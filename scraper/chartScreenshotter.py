import os
import re
from shutil import copyfile
from selenium import webdriver
from PIL import Image
from io import BytesIO
from time import sleep
import base64
import pandas as pd

scrapedPath = 'scraped/metadata.csv'
filePath = '../dataset/images/statista'

options = webdriver.ChromeOptions()
options.add_argument('--disable-notifications')
options.add_argument("headless")
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
scraped = pd.read_csv(scrapedPath)

for row in scraped.iterrows():
    if f'{row[1].id}.png' not in os.listdir(filePath):
        print(f'{row[1].id}.png')
        cleanUrl = row[1].URL
        browser.get(cleanUrl)
        sleep(1)
        try:
            element = browser.find_element_by_class_name('statisticHeader')
            location = element.location
            size = element.size
            png = browser.get_screenshot_as_png()  # saves screenshot of entire page

            im = Image.open(BytesIO(png))  # uses PIL library to open image in memory

            left = location['x']
            top = location['y']
            right = location['x'] + size['width']
            bottom = location['y'] + 760

            im = im.crop((left, top, right, bottom))  # defines crop points
            im.save(f'{filePath}/{row[1].id}.png')  # saves new cropped image
        except Exception as ex:
            print(ex)

browser.quit()