import csv
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from unidecode import unidecode
from webdriver_manager.chrome import ChromeDriverManager


URLS = [
    "https://www.quora.com/topic/International-Travel",
    "https://www.quora.com/topic/Visiting-and-Travel-1",
    "https://www.quora.com/topic/Tourism",
    "https://www.quora.com/topic/Vacations",
    "https://www.quora.com/topic/Hotels",
]


def scrape() -> None:
    """
    Scrapes Quora website for questions and answers from multiple URLs and save them to .csv files

    :return:
        None
    """
    for url in URLS:
        print(f"Scraping URL: {url}")
        for j in range(10):
            print(f"Attempt: {j}")
            chrome_options = Options()
            chrome_options.add_experimental_option("detach", True)
            chrome_options.add_argument(
                f"user-data-dir={os.getenv('CHROME_DATA_PATH')}"
            )

            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=chrome_options
            )

            driver.get(url)

            time.sleep(5)

            for i in range(100):
                print(f"Page: {i}")
                try:
                    driver.find_element(
                        By.XPATH,
                        "//div[contains(@class, 'LoadingDots___StyledFlex-sc-1r7wywh-1 cQAEU')]",
                    ).click()
                    time.sleep(5)
                except:
                    break

            counter = 0
            qna = []

            questions = driver.find_elements(
                By.XPATH,
                "//div[@class='q-box qu-mb--tiny']",
            )
            questions = [q.text for q in questions]

            answers = driver.find_elements(
                By.XPATH,
                "//div[@class='q-box spacing_log_answer_content puppeteer_test_answer_content']",
            )

            ans = []
            for a in answers:
                if "(more)" in a.text:
                    driver.execute_script("arguments[0].click();", a)
                ans.append(a.text)
            answers = ans

            driver.close()

            for q, a in zip(questions, answers):
                qna.append([counter, q, a])
                counter += 1
                print(f"Q: {q}")
                print(f"A: {a}")

            with open(
                f"../data/traveling_qna_dataset{j}{url.split('/')[-1]}.csv",
                "w",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["", "Question", "Answer"])
                writer.writerows(qna)


def combine_files() -> None:
    """
    Combines, removes duplicates and transforms unicode to ascii chars and save questions and answers to single file

    :return:
        None
    """
    data = []
    for url in URLS:
        for i in range(10):
            with open(
                f'../data/traveling_qna_dataset{i}{url.split("/")[-1]}.csv',
                "r",
                encoding="utf-8",
                newline="\n",
            ) as f:
                reader = csv.reader(f, delimiter="\t")
                data += list(reader)

    data = [(unidecode(d[1]), unidecode(d[2])) for d in data]
    data = list(set(data))
    data.remove(("Question", "Answer"))
    data = [[i, d[0], d[1]] for i, d in enumerate(data)]
    data.insert(0, ["", "Question", "Answer"])

    with open("../data/traveling_qna_dataset.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(data)


if __name__ == "__main__":
    scrape()
    combine_files()
