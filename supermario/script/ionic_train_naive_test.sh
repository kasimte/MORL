# python train.py --env-name SuperMarioBros-v0 --method naive --model test --gamma 0.99 --mem-size 4000 --batch-size 20 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 8 --episode-num 100 --optimizer Adam --save /scratch/runzhey/saved/ --log /scratch/runzhey/logs/ --update-freq 100 --name test
python train.py --env-name SuperMarioBros-v0 --method naive --model test --gamma 0.99 --mem-size 4000 --batch-size 64 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 16 --episode-num 10000 --optimizer Adam --save saved/ --log logs/ --update-freq 100 --name test_gpu