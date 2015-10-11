while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/fut.txt 
while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/finna.txt
while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/fineu.txt
while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/ase.txt
while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/hf15.txt
while read LINE
do
echo $LINE
sudo wget -O $LINE.csv "real-chart.finance.yahoo.com/table.csv?s="$LINE"&a=00&b=
3&c=2000&d=09&e=30&f=2015&g=d&ignore=.csv"
done < /usr/local/lib/python2.7/dist-packages/QSTK-0.2.8-py2.7.egg/QSTK/QSData/Yahoo/Lists/telcos.txt
