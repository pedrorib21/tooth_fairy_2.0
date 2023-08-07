train:
	python train.py --epochs=10 --teeth_to_identify=LeftMandibularSecondMolar

evaluate:
	python evaluate.py --teeth_to_identify=LeftMandibularSecondMolar -mp "data/tmp/models/21_11_42_07_08_2023/"