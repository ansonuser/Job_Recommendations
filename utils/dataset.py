from typing import Dict

class Data:
    def __init__(self, **kwarg:Dict[str, str]):
        self.features = list(kwarg.keys())
        for k,v in kwarg.items():
            setattr(self, k.replace(" ", "_"), v)

    def form(self):
        article = ""
        for f in self.features:
            article += f
            article += ":" + getattr(self, f)
            article += "\n"
        return article

class Resume(Data):
    def __init__(self, **kwargs: Dict[str, str]):
        super().__init__(**kwargs)
        self.class_type = "Resume"  

class Job(Data):
    def __init__(self, **kwargs: Dict[str, str]):
        super().__init__(**kwargs)
        self.class_type = "Job"  