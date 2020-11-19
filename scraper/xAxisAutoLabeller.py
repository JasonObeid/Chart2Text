import os
import pandas as pd

states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','MontanaNebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','PennsylvaniaRhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
countries = ['Afghanistan','Albania','Algeria','Andorra','Angola','Antigua and Barbuda','Argentina','Armenia','Australia','Austria','Azerbaijan','Bahamas','Bahrain','Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bhutan','Bolivia','Bosnia','and','Herzegovina','Botswana','Brazil','Brunei','Bulgaria','Burkina','Faso','Burundi','Cabo','Verde','Cambodia','Cameroon','Canada','Central','African','Republic','(CAR)','Chad','Chile','China','Colombia','Comoros','Congo','','Democratic','Republic','of','the','Congo','Republic','of','the','Costa','Rica','Cote','dIvoire','Croatia','Cuba','Cyprus','Czechia','Denmark','Djibouti','Dominica','Dominican Republic','Ecuador','Egypt','El Salvador','Equatorial','Guinea','Eritrea','Estonia','Eswatini','(formerly','Swaziland)','Ethiopia','Fiji','Finland','France','Gabon','Gambia','Georgia','Germany','Ghana','Greece','Grenada','Guatemala','Guinea','Guinea-Bissau','Guyana','Haiti','Honduras','Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati','Kosovo','Kuwait','Kyrgyzstan','Laos','Latvia','Lebanon','Lesotho','Liberia','Libya','Liechtenstein','Lithuania','Luxembourg','M','Madagascar','Malawi','Malaysia','Maldives','Mali','Malta','Marshall','Islands','Mauritania','Mauritius','Mexico','Micronesia','Moldova','Monaco','Mongolia','Montenegro','Morocco','Mozambique','Myanmar','Burma','Namibia','Nauru','Nepal','Netherlands','New','Zealand','Nicaragua','Niger','Nigeria','North','Korea','North Macedonia','Macedonia','Norway','Oman','Pakistan','Palau','Palestine','Panama','Papua New Guinea','Paraguay','Peru','Philippines','Poland','Portugal','Qatar','Romania','Russia','Rwanda','Saint','Kitts','and','Nevis','Saint','Lucia','Saint','Vincent','and','the','Grenadines','Samoa','San','Marino','Sao','Tome','and','Principe','Saudi','Arabia','Senegal','Serbia','Seychelles','Sierra','Leone','Singapore','Slovakia','Slovenia','Solomon','Islands','Somalia','South','Africa','South','Korea','South','Sudan','Spain','Sri','Lanka','Sudan','Suriname','Sweden','Switzerland','Syria','Taiwan','Tajikistan','Tanzania','Thailand','Timor-Leste','Togo','Tonga','Trinidad','and','Tobago','Tunisia','Turkey','Turkmenistan','Tuvalu','Uganda','Ukraine','United Arab Emirates (UAE)','United','Kingdom','(UK)','United','States','of','America','(USA)','Uruguay','Uzbekistan','Vanuatu','Vatican','City','Venezuela','Vietnam','Yemen','Zambia','Zimbabwe']
Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'November', 'December']
mnths = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = []
yrs = []
platforms = ['YouTube','Facebook','WhatsApp','Instagram','Facebook Messenger','Twitter','Pinterest','Taringa','LinkedIn','Skype','Snapchat']
sex = ['Male', 'Female']
for i in range(1850, 2051):
    years.append(str(i))
for i in range(7, 22):
    yrs.append(str(i))
    yrs.append('\''+str(i))
Quarters = ['Q1', 'Q2', 'Q3', 'Q4']
fileList = os.listdir('2Columns')

for filePath in fileList:
    df = pd.read_csv('2Columns/'+filePath)
    xAxis = df.columns[0]
    yAxis = df.columns[1]
    #automate for: year(yyyy, yy, 'yy), quarters, months, countries, states
    if(xAxis == 'Unnamed: 0'):
        print(filePath)
        xColumn = df[xAxis]
        newXAxis = ''
        if yAxis == 'Share of respondents':
            newXAxis = 'Response'
            print('found a survey')
        for value in xColumn:
            if value in states:
                newXAxis = 'State'
                print('found a state')
                break
            elif value in countries:
                newXAxis = 'Country'
                print('found a country')
                break
            elif str(value)[0:4] in years:
                newXAxis = 'Year'
                print('found a year')
                break
            elif str(value)[0:2] in yrs:
                newXAxis = 'Year'
                print('found a year')
                break
            elif value[0:2] in Quarters:
                newXAxis = 'Quarter'
                print('found a quarter')
                break
            elif value in Months:
                newXAxis = 'Month'
                print('found a month')
                break
            elif value[0:3] in mnths:
                newXAxis = 'Month'
                print('found a month')
                break
            elif value in sex:
                newXAxis = 'Sex'
                print('found a sex')
                break
            elif value in platforms:
                newXAxis = 'Platform'
                print('found a platform')
                break
        df.columns = [newXAxis, df.columns[1]]
        print(df.head())
        df.to_csv(index=False, path_or_buf=('2Columns/'+filePath))

