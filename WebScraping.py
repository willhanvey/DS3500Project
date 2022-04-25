import pandas as pd
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os


PYCHARM_DIR = "C:\\Users\\baseb\\PycharmProjects\\DS3500Project\\DraftResolutions"
SERIUM_CHROME_PATH = "C:/Users/baseb/Downloads/chromedriver.exe"


def open_file(file):
    """
    :param file: File to read into a dataframe
    :return: Dataframe
    """
    df = pd.read_csv(file, low_memory=False)
    return df


def selenium_setup():
    """
    :return: Options setup to be used for selenium
    """
    # Configures selenium
    options = webdriver.ChromeOptions()
    profile = {"plugins.plugins_list": [{"enabled": False, "name": "Chrome PDF Viewer"}],  # Disable Chrome's PDF Viewer
               "download.default_directory": PYCHARM_DIR, "download.extensions_to_open": "applications/pdf"}
    options.add_experimental_option("prefs", profile)
    return options


def un_webscrape(df, options):
    """
    :param df: dataframe with links
    :param options: options from selenium setup func
    :return: none
    Downloads all the pdfs from the links in the dataframe
    """
    counter = 0
    pdf_list = []
    error_list = []
    for i in range(len(df.index)):
        driver = webdriver.Chrome(executable_path= SERIUM_CHROME_PATH, chrome_options=options)
        link = df.iloc[i][9]
        driver.get(link)
        try:
            # Clicks the buttons from the UN webpage to download the files
            l = driver.find_element(By.XPATH, '//*[@id="details-collapse"]/div[5]/span[2]/a')
            l.click()
            l = driver.find_element(By.XPATH, '//*[@id="record-files-list"]/tbody/tr[2]/td[1]/a')
            l.click()
            num_files = len([f for f in os.listdir('.\\DraftResolutions') if f.endswith('.pdf')])
            pdf_list.append((driver.find_element(By.XPATH, '//*[@id="record-files-list"]/tbody/tr[2]/td[2]')).text)
            time.sleep(.5)
            # Checks if the pdf has been added to the target location, if not, waits and tries again
            while num_files == len([f for f in os.listdir('.\\DraftResolutions') if f.endswith('.pdf')]):
                time.sleep(.1)
        except selenium.common.exceptions.NoSuchElementException:
            pdf_list.append('Error')
            error_list.append(counter)
        print(counter)
        counter += 1
        driver.quit()
    return pdf_list, error_list


def write_to_file(pdf_list, error_list):
    """
    :param pdf_list: list of pdfs
    :param error_list: list of positions of errors
    :return: None
    Writes pdf_lists and error_lists to files for later use
    """
    with open('PDFList.txt', 'w') as infile:
        for item in pdf_list:
            infile.write(str(item) + '\n')
    with open('ErrorList.txt', 'w') as infile:
        for item in error_list:
            infile.write(str(item) + '\n')


def main():
    df = open_file('UN DATA.csv')
    options = selenium_setup()
    pdf_list, error_list = un_webscrape(df, options)
    write_to_file(pdf_list, error_list)
    print('Complete!')


if __name__ == "__main__":
    main()
