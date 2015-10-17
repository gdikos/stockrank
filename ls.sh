#!/bin/bash

COUNTER=0
for style in 0 1 2
do
    for i in 2 10
    do
       for holding_period in 5 10
       do
       for trigger in -0.10 -0.05 0.05 0.10 
       do
       for market in 0.02 0 0.02 
           do
           for bin in -1 1
           do 
             for switch in -1 1
             do
    echo "lookback period is" $i >> test2.txt   
    echo "holding period is" $holding_period >> test2.txt 
    echo "trigger is" $trigger >> test2.txt    
    echo "lookback period is" $i 
    echo "holding period is" $holding_period 
    echo "trigger is" $trigger
    echo "market band is" $market
if [ "$COUNTER" -gt "390" ]
then echo "counter is" $COUNTER >> sharpe_b2.txt 
    sudo python ls.py $i $holding_period $trigger $market $bin $switch $style>> sharpe_b2.txt 
#   sudo python bollinger_events.py $i $holding_period $trigger $market $bin $switch >> score.txt

    sudo python ls_sim.py
#   sudo python analyzer.py $i $holding_period $trigger $market $bin $switch
    echo "counter is" $COUNTER >> sharpe_r2.txt
    sudo python ls_analyzer.py $i $holding_period $trigger $market $bin $switch >> sharpe_r2.txt 
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
