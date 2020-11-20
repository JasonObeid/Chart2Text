import json
import re

import pandas as pd
import numpy as np
import os
import PIL as pil
import matplotlib.pyplot as plt

def getChartType(x):
    if x.lower() == 'year':
        return 'line_chart'
    else:
        return 'bar_chart'


complexDataList = os.listdir('../dataset/multiColumn/data/')
dataList = os.listdir('../dataset/data/')

dataList.sort()
complexDataList.sort()

for dataPath in dataList:
    try:
        if f'{dataPath[:-4]}.png' not in os.listdir(f'../dataset/images/matplot'):
            df = pd.read_csv('../dataset/data/' + dataPath)
            chartType = getChartType(df.columns[0])
            dico = {label: 'float32' for label in df.columns[1:]}
            imgPath = f'../dataset/images/matplot/{dataPath[:-4]}.png'
            if chartType == 'bar_chart':
                try:
                    df = df.astype(dico)
                except Exception as e:
                    for n in df.columns[1:]:
                        for m in df.index:
                            value = df.at[m, n]
                            newValue = re.sub("[^\d\.]", "", value)
                            if newValue != '':
                                df.at[m, n] = float(newValue)
                            else:
                                df.at[m, n] = 0
                
                df.set_index(df.columns[0], drop=True, inplace=True)
                ax = df.plot.bar()
            else:
                try:
                    df = df.astype(dico)
                except Exception as e:
                    for n in df.columns:
                        for m in df.index:
                            value = str(df.at[m, n])
                            if value != '':
                                newValue = re.sub("[^\d\.]", "", value)
                                if newValue != '':
                                    newValue = newValue.replace('*','')
                                    df.at[m, n] = float(newValue)
                                else:
                                    df.at[m, n] = 0
                            else:
                                df.at[m, n] = 0
                df.set_index(df.columns[0], drop=True, inplace=True)
                ax = df.plot.line()
            with open(f'../dataset/titles/{dataPath[:-4]}.txt') as titleFile:
                plt.title(titleFile.read().strip())
            ax.set_ylabel(df.columns[0])
            plt.savefig(imgPath, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(dataPath)

for complexDataPath in complexDataList:
    try:
        imgPath2 = f'{complexDataPath[:-4]}.png'
        if imgPath2 not in os.listdir('../dataset/multiColumn/images/matplot'):
            df = pd.read_csv('../dataset/multiColumn/data/' + complexDataPath)
            dico = {label: 'float32' for label in df.columns[1:]}
            imgPath = f'../dataset/multiColumn/images/matplot/{imgPath2}'
            try:
                df = df.astype(dico)
            except Exception as e:
                for n in df.columns[1:]:
                    for m in df.index:
                        value = str(df.at[m, n])
                        if value != '' and value != '-':
                            newValue = re.sub("[^\d\.]", "", value)
                            if newValue != '' and newValue != '-':
                                newValue = newValue.replace('*','')
                                df.at[m, n] = float(newValue)
                            else:
                                df.at[m, n] = 0
                        else:
                            df.at[m, n] = 0
            df.set_index(df.columns[0], drop=True, inplace=True)
            chartType = getChartType(df.index.name)
            if chartType == 'bar_chart':
                ax = df.plot.bar()
            else:
                ax = df.plot.line()
                plt.xticks(rotation='vertical')
                # Pad margins so that markers don't get clipped by the axes
                plt.margins(0.05)
                # Tweak spacing to prevent clipping of tick-labels
                plt.subplots_adjust(bottom=0.1)
            with open(f'../dataset/multiColumn/titles/{complexDataPath[:-4]}.txt') as titleFile:
                plt.title(titleFile.read().strip())
            ax.set_xlabel(df.index.name)
            plt.savefig(imgPath, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(e)
        print(complexDataPath)