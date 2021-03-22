import urllib
from bs4 import BeautifulSoup
import csv
import os
import time
import click
import pandas as pd

"""
echo "carat,cut,color,clarity,table,depth,cert,measurements,price,wlink" >> Diamonds_100817.csv
cat data/*.csv > Diamonds_tmp.csv
grep -vwE "carat" Diamonds_tmp.csv >> Diamonds_100817.csv
rm Diamonds_tmp.csv
"""

@click.command()
@click.option("--resume", type=bool, default=False, is_flag=True)
@click.option("--outdir", type=str, default='data')

def main(resume, outdir):

    os.system('mkdir -p {}'.format(outdir))
    print('Creating directory {}'.format(outdir))

    shape = 'Round'
    minCarat = '0.2'
    maxCarat = '2'
    minColor = '1'
    maxColor = '9'
    minPrice = '1'
    maxPrice = '10000'
    minCut = '5'
    maxCut = '1'
    minClarity = '1'
    maxClarity = '10'
    minDepth = '0'
    maxDepth = '90'
    minWidth = '0'
    maxWidth = '90'
    gia = '1'
    ags = '1'
    egl = '1'
    oth = '1'
    currency = 'USD'
    rowStart = '0'
    sortCol = 'price'
    sortDir = 'ASC'
    thisrow = 0

    # about 26k pages
    # about 598k diamonds
    # chunk by about 2.5k diamonds
    # 598000/2500 = 239

    thischunk = 0
    thisrow =  thischunk * 2500
    rowStart = str(thisrow)
    thiscount = 0

    df = pd.DataFrame()

    while (thischunk < 110):
        try:
            #f = csv.writer(open(outdir + "/" + str(thischunk) + ".csv", "wb"))
            # Write column headers as the first line
            # f.writerow(["carat", "cut", "color", "clarity", "table", "depth", "cert", "measurements", "price", "wlink"])
            while (thiscount < 2500):
                uri = "http://www.diamondse.info/webService.php?shape="+shape+"&minCarat="+minCarat+"&maxCarat="+maxCarat+"&minColor="+minColor+"&maxColor="+maxColor+"&minPrice="+minPrice+"&maxPrice="+maxPrice+"&minCut="+minCut+"&maxCut="+maxCut+"&minClarity="+minClarity+"&maxClarity="+maxClarity+"&minDepth="+minDepth+"&maxDepth="+maxDepth+"&minWidth="+minWidth+"&maxWidth="+maxWidth+"&gia="+gia+"&ags="+ags+"&egl="+egl+"&oth="+oth+"&currency="+currency+"&rowStart="+rowStart+"&sortCol="+sortCol+"&sortDir="+sortDir
                print(uri)
                urllines = urllib.urlopen(uri)
                print(urllines)
                pagedat = urllines.read()
                print(pagedat)
                urllines.close()
                soup = BeautifulSoup(pagedat, "lxml")
                for row in soup.find_all("tr"):
                    tds = row.find_all("td")
                    print(tds)
                    try:
                        for link in tds[0].find_all('a'):
                            wlink = "http://www.diamondse.info/"+link.get('href')
                        carat = str(tds[2].get_text())
                        cut = str(tds[3].get_text())
                        color = str(tds[4].get_text())
                        clarity = str(tds[5].get_text())
                        table = str(tds[6].get_text())
                        depth = str(tds[7].get_text())
                        cert = str(tds[8].get_text())
                        measurements = str(tds[9].get_text())
                        price = str(tds[10].get_text())
                    except:
                        print("bad string")
                        time.sleep(20)
                        continue
                    #print [carat, cut, color, clarity, table, depth, cert, measurements, price]
                    #f.writerow([carat, cut, color, clarity, table, depth, cert, measurements, price, wlink])
                    df = df.append({'carat': carat,
                                    'cut': cut,
                                    'color': color,
                                    'clarity': clarity,
                                    'table': table,
                                    'depth': depth,
                                    'cert': cert,
                                    'measurements': measurements,
                                    'price': price}, ignore_index=True)
                time.sleep(0.1)
                thiscount = thiscount + 20
                thisrow = thisrow + 20
                rowStart = str(thisrow)
                print('this row = ' + rowStart)
            thischunk = thischunk + 1
            thiscount = 0
        except:
            print("Possible connectivity issues")
            time.sleep(10)
            thisrow =  thischunk * 2500
            rowStart = str(thisrow)
            continue



if __name__ == "__main__":
    main()
