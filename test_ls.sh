#!/bin/bash

COUNTER=0
for style in -42 -22 -12 -6 -5 -4 -3 -2 -1 2
do
    for i in 1 2 10 30 
    do
       for holding_period in 5 10 30 
       do
       for trigger in 2  
       do
       for market in 0
           do
           for bin in -1 1 
           do 
             for switch in 0 
             do
		for sharpe_period in 10 30 90 300
                do
			for quantile in 5 7 8 9 
			do
    echo "lookback period is" $i >> test2.txt   
    echo "holding period is" $holding_period >> test2.txt 
    echo "trigger is" $trigger >> test2.txt    
    echo "lookback period is" $i 
    echo "holding period is" $holding_period 
    echo "trigger is" $trigger
    echo "market band is" $market
    echo "sharpe_period" $sharpe_period
    echo "quantile" $quantile
    echo $COUNTER

if [ "$COUNTER" -gt "0" ]
#then echo "counter is" $COUNTER >> sims/sharpe_b2.txt 
#    sudo python ls.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile>> sims/sharpe_b2.txt 

#    sudo python ls_sim.py
#    echo "counter is" $COUNTER >> sims/sharpe_r2.txt
#    sudo python ls_analyzer.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile $COUNTER >> sims/sharpe_r2.txt 
then sudo python ls.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile && python ls_sim.py && python ls_analyzer.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile $COUNTER >> sims/merged.txt
fi
COUNTER=$((COUNTER+1))
echo $COUNTER
done
done
done
done
done
done
done
done
done
