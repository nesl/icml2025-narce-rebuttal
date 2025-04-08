for dataset in 2000 4000; do # 1000 2000 4000 6000 8000 10000
    for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $dataset
        echo $seed
        python main_ced.py --no_flow --dataset $dataset --seed $seed 
        python main_ced.py --no_flow --dataset $dataset --seed $seed --eval True --testset '15min'  
        python main_ced.py --no_flow --dataset $dataset --seed $seed --eval True --testset '30min' 
    done
done

