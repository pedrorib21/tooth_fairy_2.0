train:
	python train.py --epochs=100 --teeth_to_identify=LeftMandibularSecondMolar --teeth_to_identify RightMandibularFirstMolar

evaluate:
	python evaluate.py --teeth_to_identify=LeftMandibularSecondMolar -mp "data/tmp/models/23_07_31_06_08_2023/"