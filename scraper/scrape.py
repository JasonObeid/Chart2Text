import re
import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup

sr = []
txt_dir = "captions/"
base_dir = "2Columns/"
other_dir = "multiColumn/"
metaPath = base_dir + "metadata.csv"
otherMetaPath = other_dir + "otherMetadata.csv"
meta_list = []
other_meta_list = []

class MySpider(scrapy.Spider):
    name = "ScraperWithLimit"
    start_urls = []
    for i in range(0, 1000):
        start_urls.append('https://www.statista.com/statistics/popular/p/' + str(i) + '/')
    print(start_urls)
    allowed_domains = ['statista.com/markets/', 'statista.com/statistics/']
    custom_settings = {
        'DEPTH_LIMIT': 2
    }

    def parse(self, response):
        le = LinkExtractor()
        for link in le.extract_links(response):
            if (link not in sr):
                url = link.url
                if ('/statistics/' in url):
                    #ignore index pages i.e 'https://www.statista.com/statistics/popular/p/93/'
                    #length of 51 covers index page up to 1000
                    if(len(url) > 51):
                        #print(url)
                        sr.append(link)
                yield response.follow(link, self.parse)


def processData(captionText, dFrame, link, ids):
    # save dataframe only if it has 2 columns
    dataPath = base_dir + str(ids) + ".csv"
    xAxis = dFrame.columns[0]
    yAxis = dFrame.columns[1]
    if (xAxis.lower() == 'year'):
        chartType = "line"
    else:
        chartType = 'bar'
    # list of metadata contains item id, path of data.csv, and the caption
    meta_list.append({'id': ids, 'dataPath': dataPath, 'caption': captionText, 'chartType': chartType, 'xAxis': xAxis,
                      'yAxis': yAxis, 'title': link.text, 'URL': link.url})
    dFrame.to_csv(index=False, path_or_buf=dataPath)
    txtPath = txt_dir + str(ids) + ".txt"
    with open(txtPath, 'w') as f:
      f.write(captionText)


def processMultiColumn(captionText, dFrame, link, multiIds):
    dataPath = other_dir + str(multiIds) + ".csv"
    xAxis = dFrame.columns
    columns = []
    for row in xAxis:
        columns.append(row)
    # list of metadata contains item id, path of data.csv, and the caption
    other_meta_list.append({'id': multiIds, 'dataPath': dataPath, 'caption': captionText, 'xAxis': columns, 'URL': link.url})
    dFrame.to_csv(index=False, path_or_buf=dataPath)
    txtPath = other_dir + 'captions/' + str(multiIds) + ".txt"
    with open(txtPath, 'w') as f:
      f.write(captionText)


def crawl():
    sr = []
    process = CrawlerProcess()
    process.crawl(MySpider)
    process.start()
    process.stop()
    print(sr)
    print(len(sr))

    Path = "scraped.csv"
    frame = pd.DataFrame(sr)
    frame.to_csv(index=False, path_or_buf=Path)

    return sr


def scrape(sr):
    ids = 0
    multiIds = 0
    for link in sr:
        # open link
        html = urlopen(link.url).read()
        # parse as soup object
        soup = BeautifulSoup(html, 'html.parser')
        #print(soup)
        if (soup.body.find("table")):
            # find data tables in soup object
            caption = ""
            tableList = soup.body.find_all(id = 'statTable')
            for table in tableList:
                #convert html to dataframe
                dfs = pd.read_html(str(table))[0]
                #check if table is empty
                if(dfs.iloc[0][0] != '-'):
                    # take caption
                    textList = soup.body.find_all(class_='responsiveText readingAid__text')
                    for text in textList:
                        #remove /n, /r, /t
                        regex = re.compile(r'[\n\r\t]')
                        cleanText = regex.sub(" ", text.get_text())
                        #remove extra whitespace
                        cleanText = re.sub("\s\s+", " ", cleanText)
                        caption += cleanText
                    #check how many columns table has
                    if (dfs.shape[1] == 2):
                        processData(caption, dfs, link, ids)
                        ids += 1
                    else:
                        processMultiColumn(caption, dfs, link, multiIds)
                        multiIds += 1

    metadata = pd.DataFrame(meta_list)
    metadata.to_csv(index=False, path_or_buf=metaPath)

    otherMeta = pd.DataFrame(other_meta_list)
    otherMeta.to_csv(index=False, path_or_buf=otherMetaPath)

try:
    urls = crawl()
    scrape(urls)

except Exception as ex:
    print(ex)