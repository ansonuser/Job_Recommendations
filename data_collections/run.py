import os
import sys
sys.path.append(os.getcwd() + f"{os.sep}..")
from muse_scraper import TheMuse
from dice_scraper import Dice


def crawling():
    keywords = ["Data", "Scientist", "Machine"]
    scraper = TheMuse(last_days=14,limit_per=100)
    _ = scraper.search(keywords)
    scraper.close_elastic()

    dice = Dice()
    dice.run(keywords, 100)


if __name__ == "__main__":
    crawling()