from render import Raytracer, V3, V2, color, norm, PI
from utils import color
from obj import Obj, Texture, Envmap
from sphere import *

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


width = 800
height = 800
r = Raytracer(width, height)
r.glClearColor(0.2, 0.6, 0.8)
r.glClear()

r.envmap = Envmap('envmap.bmp')

r.dirLight = DirectionalLight(direction = V3(1, -1, -2), intensity = 0.5)
r.ambientLight = AmbientLight(strength = 0.1)

brick = Material(diffuse = color(0.8, 0.25, 0.25 ), spec = 16)
stone = Material(diffuse = color(0.4, 0.4, 0.4 ), spec = 32)
grass = Material(diffuse = color(0.5, 1, 0), spec = 32)
glass = Material(spec = 64, ior = 1.5, matType= TRANSPARENT) 
snow = Material(diffuse= color(1,1,1))
dot = Material(diffuse= color(0,0,0))
carrot = Material(diffuse= color(1,0.6,0))
mirror = Material(spec = 64, matType = REFLECTIVE)
water = Material(diffuse=color(0.6,1,1), spec = 100, ior = 3, matType= TRANSPARENT)
enderPearl = Material(diffuse=color(0,0.3,0.3), spec = 40, ior = 1.5, matType= OPAQUE)

sky = Material(diffuse= color(0.6,0.8,0.9))


sandMat = Material(texture = Texture('sand.bmp'))


# r.pointLights.append(PointLight(position = V3(2,-1,0), intensity = 1))
#r.pointLights.append(PointLight(position = V3(2,1,0), intensity = 1))
r.ambientLight = AmbientLight(strength = 0.1)


noisemap = Texture('noiseMap.bmp')

for i in range(-15,15):
    for j in range(1,50):
        if noisemap.pixels[i][j][0]/ 255 >=0.66:
            r.scene.append(AABB(V3((i*0.5),-2, (-2-j*0.5)), V3(0.5,0.5,0.5), water))
        if noisemap.pixels[i][j][0] / 255 < 0.66 and noisemap.pixels[i][j][0] / 255 >=0.33:
            r.scene.append(AABB(V3((i*0.5),-1.5, (-2-j*0.5)), V3(0.5,0.5,0.5), sandMat))
        else:
            r.scene.append(AABB(V3((i*0.5),-2.5, (-2-j*0.5)), V3(0.5,0.5,0.5), snow))
            

r.scene.append(AABB(V3(-0.6, 0,-2), V3(0.5,0.5,0.5), mirror))
r.scene.append( Sphere(V3(4, 0, -10),  1, enderPearl) )
r.scene.append( Sphere(V3(2, 0, -5),  0.5, glass) )

# r.scene.append(AABB(V3(-0.6, 0,-2), V3(0.5,0.5,0.5), water))


r.rtRender()


r.glFinish('output.bmp')

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
