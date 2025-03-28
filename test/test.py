import time
from tqdm import tqdm


pbar = tqdm(total=100, desc="Converting data", unit="file")
update = lambda *args: pbar.update()

for i in range(100):
   update()
   time.sleep(0.1)