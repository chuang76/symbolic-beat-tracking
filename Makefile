all: run clean 

run:
	@echo "Preprocessing..."
	@python pre-process.py 
	@echo "Running..."
	@python run.py
	@python post-process.py 

clean:
	@rm -r ./input/raw 
	@rm -r ./input/npz
	@rm -r tmp 



