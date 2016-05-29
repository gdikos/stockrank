sort tmpfile2.txt | uniq >  tmpfile3.txt
sort -t ',' -k1,1 -k2,2 -k3,3n tmpfile.csv >> orderbook.csv
