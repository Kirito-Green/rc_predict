mkdirs:
	mkdir -p ../data/raw_data/
	mkdir -p ../data/convert_data/
	mkdir -p ../data/graph_data/
	mkdir -p ../params/
	mkdir -p ../results/
	mkdir -p ../logs/

clean:
	rm -rf ../params/
	rm -rf ../results/
	rm -rf ../logs/

run:
	./run.sh
