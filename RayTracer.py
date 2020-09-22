from render import Raytracer, V3, V2, color
from utils import color
from obj import Obj, Texture
from sphere import Sphere, Material, PointLight, AmbientLight

brick = Material(diffuse = color(0.8, 0.25, 0.25 ), spec = 16)
stone = Material(diffuse = color(0.4, 0.4, 0.4 ), spec = 32)
grass = Material(diffuse = color(0.5, 1, 0), spec = 32)
glass = Material(diffuse = color(0.25, 1, 1), spec = 64)
snow = Material(diffuse= color(1,1,1))
dot = Material(diffuse= color(0,0,0))
carrot = Material(diffuse= color(1,0.6,0))

width = 1080
height = 720
r = Raytracer(width, height)

r.pointLight = PointLight(position = V3(-2,2,0), intensity = 1)
r.ambientLight = AmbientLight(strength = 0.1)

r.scene.append( Sphere(V3(    0,   0, -5),    1, brick) )
r.scene.append( Sphere(V3( -0.5, 0.5, -3), 0.25, stone) )
#r.scene.append( Sphere(V3(-1,-1, -5), 0.5, grass) )
#r.scene.append( Sphere(V3( 1,-1, -5), 0.5, glass) )
 
r.rtRender()




r.glFinish('output.bmp')
