#!/bin/bash
echo -n "Enter start date > "
read s_beg
echo -n "Enter end date > "
read s_end
echo -n "Enter asset class > "
read asset_class
echo -n "Enter index > "
read index
echo -n "Enter counter cut-off > "
read cutoff
echo -n "You entered > "
echo -n $s_beg $s_end $asset_class $index $cutoff
COUNTER=0
for style in -42 -32 -22 -12 -6 -5 -4 -3 -2 -1 2 22 32 42 201 203 232
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
             for switch in -1 0 1 
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

if [ "$COUNTER" -gt $cutoff ]
#then echo "counter is" $COUNTER >> sims/sharpe_b2.txt 
#    sudo python ls.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile>> sims/sharpe_b2.txt 

#    sudo python ls_sim.py
#    echo "counter is" $COUNTER >> sims/sharpe_r2.txt
#    sudo python ls_analyzer.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile $COUNTER >> sims/sharpe_r2.txt 
then sudo python ls.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile $s_beg $s_end $asset_class $index $COUNTER >> sims/sims/log_$asset_class()_$index$()_arbtrade_$s_beg$()_$s_end$().txt && python ls_sim.py && python ls_analyzer.py $i $holding_period $trigger $market $bin $switch $style $sharpe_period $quantile $asset_class $index $COUNTER>> sims/alpha_$asset_class()_$index$()_arbtrade_$s_beg$()_$s_end$().txt
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
