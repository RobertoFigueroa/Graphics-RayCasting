from render import Raytracer, V3, V2, color, norm, PI
from utils import color
from obj import Obj, Texture, Envmap
from sphere import *

width = 256
height = 256
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
rock = Material(diffuse=color(0.5,0.5,0.5), spec = 30, ior = 1.5, matType= OPAQUE)
sky = Material(diffuse= color(0.6,0.8,0.9))

boxMat = Material(texture = Texture('box.bmp'))

earthMat = Material(texture = Texture('earthDay.bmp'))

r.pointLight = PointLight(position = V3(0,0,0), intensity = 1)
r.ambientLight = AmbientLight(strength = 0.1)


noisemap = Texture('noiseMap.bmp')

for i in range(1,20):
    for j in range(1,20):
        if noisemap.pixels[i][j][0]/ 255 >=0.66:
            r.scene.append(AABB(V3((i-10),0.5, -j*0.5), V3(1,1,1), boxMat))
        if noisemap.pixels[i][j][0] / 255 < 0.66 and noisemap.pixels[i][j][0] / 255 >=0.33:
            r.scene.append(AABB(V3((i-10),0, -j*0.5), V3(1,1,1), boxMat))
        else:
            r.scene.append(AABB(V3((i-10),-0.5, -j*0.5), V3(1,1,1), boxMat))
            



# r.scene.append( AABB(V3(0, 0.22, -10), V3(0.5, 0.5, 0.5) , brick ) )

# r.scene.append( Sphere(V3( 0, 0, -8), 2, earthMat))



r.rtRender()


r.glFinish('output.bmp')
