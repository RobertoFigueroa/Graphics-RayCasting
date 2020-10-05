from render import Raytracer, V3, V2, color, norm, PI
from utils import color
from obj import Obj, Texture, Envmap
from sphere import *

width = 256
height = 256
r = Raytracer(width, height)
r.glClearColor(0.2, 0.6, 0.8)
r.glClear()

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

r.pointLight = PointLight(position = V3(0,0,0), intensity = 1)
r.ambientLight = AmbientLight(strength = 0.1)

# r.scene.append( Sphere(V3(    0,   0, -5),    1, brick) )
# r.scene.append( Sphere(V3( -0.5, 0.5, -3), 0.25, stone) )
#r.scene.append( Sphere(V3(-1,-1, -5), 0.5, grass) )
#r.scene.append( Sphere(V3( 1,-1, -5), 0.5, glass) )

r.scene.append( Plane( V3(-5,-3,0), V3(1,0,0), brick))
r.scene.append( Plane( V3(5,-3,0), V3(-1,0,0), brick))
r.scene.append( Plane( V3(-2,-3,0), V3(0,1,0), rock))
r.scene.append( Plane( V3(-2,-3,-20), V3(0,0,1), sky))
r.scene.append( Plane( V3(-2,3,0), V3(0,-1,0), grass))

r.scene.append( AABB(V3(1, 1.5, -5), 1.5, stone ) )
r.scene.append( AABB(V3(-1, -1.2, -5), 1.5, glass ) )
r.scene.append( AABB(V3(1, -1.2, -5), 1.5, mirror ) )


r.rtRender()


r.glFinish('output.bmp')
