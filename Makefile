all_gnn: clean mkdirs run_gnn

all_base: clean mkdirs run_base

mkdirs:
	mkdir -p ../params/ \
	mkdir -p ../results/ \
	mkdir -p ../logs/ \
	mkdir -p ../tensorboard/

clean:
	rm -rf ../params/ \
	rm -rf ../results/ \
	rm -rf ../logs/ \
	rm -rf ../tensorboard/

run_gnn:
	./run_gnn.sh

run_base:
	./run_base.sh
