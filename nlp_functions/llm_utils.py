from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from torch.nn import BCEWithLogitsLoss, Softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import List


class BinaryClassificationLogitsProcessor(LogitsProcessorList):
    def __init__(self, class_tokens=None):
        """
        class_tokens: Dictionary mapping class labels to token IDs.
        Example: {0: 1234, 1: 5678} (Token IDs for "No" and "Yes")
        """
        self.class_tokens = class_tokens
    
    def __call__(self, input_ids, scores):
        """
        Modifies logits to allow only specific tokens (for binary classification).
        """

        new_scores = torch.full_like(scores, fill_value=-float("Inf"))

        for _, token_id in self.class_tokens.items():
            new_scores[:, token_id] = scores[:, token_id]
        
        return new_scores

class Agent:
    """
    This agent has two features.
    1. Bullet point the job description.
    2. Grade the matching score of resume and bullet point
    """
    
    def __init__(self, resume:str=None, model_path:str=None, train_epoches=5):
        self.model_path = model_path 
        self.model = None
        self.tokenizer = None
        self.resume = resume
        self.class_tokens = None
        self.train_epoches = train_epoches

    def bullet_points_prompt(self, job_description:str=None)->str:
        compress_prompt = f"""
        The content is: {job_description}
        Tell me what's your main task, what you can do and summarize in 6 bullet points around 200 words:

        """
        return compress_prompt
    
    def summarize_description(self, job_description:str=None)->str:
        prompt = f"""Base on {job_description}, summarize in 80 words, including which field it is, what's the main task:

                """
        return prompt
    
    def grade_resume_prompt(self, job_description:str=None, resume:str=None)->str:
        score_prompt = f"""
        As a headhunter, you should filter applications for company to interview candidates who matches this position most. The information for this position is {job_description}. Got a resume: {resume}.
        This resume matchs the position, 'Yes' or 'No' ? 
        """
        return score_prompt


    def load_models(self):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8",  # ✅ FP4 Quantization
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token # pad_token was undefined
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation="eager",
            # attn_implementation="flash_attention_2", 
            quantization_config=bnb_config,
            device_map="auto" 
        )

        self.tokenizer = tokenizer
        self.model = model
        self.class_tokens = { idx : self.tokenizer(text, add_special_tokens=False)["input_ids"][0] for idx, text in enumerate(["No", "Yes"])}
    def do_bullet(self, job_description:str=None)->str:
        remove_len = int(len(job_description)*1/3)
        input_text = self.bullet_points_prompt(job_description[:-remove_len])
        tokenized_inputs = self.tokenizer(input_text, return_tensors="pt")
        tokenized_inputs = {k: v.to("cuda") for k, v in tokenized_inputs.items()}
        outputs = self.model.generate(
            tokenized_inputs["input_ids"],
            # tokens marked 0 in attention mask are set to -infity in attention
            attention_mask=tokenized_inputs["attention_mask"], 
            do_sample = True,
            temperature = 0.7,
            top_k=50,
            max_new_tokens = 200
        )
        generated_tokens = outputs[0][tokenized_inputs["input_ids"].shape[1]:]
        if len(generated_tokens) == 0:
            print("Warning: No new tokens were generated.")
       
        bullet_points = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return bullet_points
    
    def get_grade_inputs(self, job_descriptions):
        job_descriptions = [self.do_bullet(job_description) for job_description in job_descriptions]
        score_prompts = [self.grade_resume_prompt(job_description, self.resume) for job_description in job_descriptions]
        tokenized_inputs = self.tokenizer(score_prompts, return_tensors="pt", padding=True)
        return tokenized_inputs

    def do_grade(self, job_descriptions:List=[], model=None):
        """
        Returns:
            torch.Tensor: Modified logits where only class tokens are allowed.
        """
        logits_processor = LogitsProcessorList([
            BinaryClassificationLogitsProcessor(self.class_tokens)
        ])
 
        tokenized_inputs = self.get_grade_inputs(job_descriptions)
        tokenized_inputs = {k: v.to("cuda") for k, v in tokenized_inputs.items()}
        if model is None:
            model = self.model
        output = model.generate(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            max_new_tokens=1,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True
        )
        logits = torch.stack(output.scores, dim=1).squeeze(1)  
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        return probs[:, self.class_tokens[1]]

    def finetuning(self, job_descriptions, labels):
        lora_config = LoraConfig(
            r=8,  # LoRA rank (low-rank adaptation)
            lora_alpha=32,  # Scaling factor (Coef of new weight: W_new = W + lora_apha*(A*B) )
            target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"  
        )
        
        peft_model = get_peft_model(self.model, lora_config)
        tokenized_inputs = self.get_grade_inputs(job_descriptions)
        tokenized_inputs = {k: v.to("cuda") for k, v in tokenized_inputs.items()}
        labels = labels.to("cuda")
        dataset = TensorDataset(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = BCEWithLogitsLoss()
        optimizer = AdamW(peft_model.parameters(), lr=2e-5) # don't know what it is...
        
        peft_model.train()
        for epoch in range(self.train_epoches):
            total_loss = 0.0
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = batch

                # Forward pass
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
               
                # decoder model is trained for predict the next token, given a size n input got a size n output, token by token.
                logits = outputs.logits[:, -1, [self.class_tokens[0], self.class_tokens[1]]]
                logits = Softmax(dim=2)(logits)

                # Compute loss
                loss = loss_fn(logits.squeeze(1)[:, 1], batch_labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.train_epoches} - Loss: {avg_loss:.4f}")



    def save_model(self, peft_model):
        save_directory = os.getcwd() + f"..{os.sep}lora_binary_classifier"
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        peft_model.save_pretrained(save_directory)
        print(f"LoRA fine-tuned model saved to {save_directory}")


