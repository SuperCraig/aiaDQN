import control as c
import random
import time

key_ground = [c.right,
              c.left,
              c.stay,
              c.up,
              c.left_p,
              c.right_p
              ]

while True:
    # modify t to make Pika more powerful(randomly)
    t = random.uniform(0,0.001)
    time.sleep(t)
    random.choice(key_ground)(1)
    