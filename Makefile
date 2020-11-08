all: run clean 

run:
	@echo "[Info] Pre-processing..."
	@python ./test/midi2csv.py 
	@python ./test/preprocess.py 

	@echo "[Info] Running..."
	@python ./test/run-blstm.py
	@python ./test/run-attn.py 
	@python ./test/run-hrnn.py

	@echo "[Info] Post-processing..."
	@python ./test/post-blstm.py
	@python ./test/post-attn.py 
	@python ./test/post-hrnn.py 

clean:
	@rm -r ./input/raw 
	@rm -r ./input/npz 
	@rm -r ./output/tmp 
