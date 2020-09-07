'''
Definition of multiple shaders
'''

from render import V3, dot, BLACK, sum, sub, norm, subV2, cross, mul, matXvect
from utils import color
from math import sin

def gourad(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']
    

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity


    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0


def toon(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)


    if intesity > 0 and intesity < 0.2:
        b *= 0.2
        g *= 0.2
        r *= 0.2
    elif intesity >= 0.2 and intesity < 0.4:
        b *= 0.4
        g *= 0.4
        r *= 0.4
    elif intesity >= 0.4 and intesity < 0.6:
        b *= 0.6
        g *= 0.6
        r *= 0.6
    elif intesity >= 0.6 and intesity < 0.8:
        b *= 0.8
        g *= 0.8
        r *= 0.8    
    elif intesity >= 0.8 and intesity < 1:
        b *= 1
        g *= 1
        r *= 1
                           
    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0


def toon_mod(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity


    if intesity > 0:
        return round(r,1), round(g,1), round(b,1)
    else:
        return 0, 0, 0




def outline(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']


    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *= intesity
    g *= intesity
    r *= intesity

    if intesity > 0 and intesity <= 0.2:
        return 1,1,1


    if intesity > 0:
        return r, g, b
    else:
        return 0, 0, 0


def unlit(render, **kwargs):
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx,ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    return r, g, b



def transparent(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']
    pixel = kwargs['pixel']

    #getting pixel backgrund color 
    color = render.framebuffer[pixel.y][pixel.x]
    b_background = color[0] / 255
    g_background = color[1] / 255
    r_background = color[2] / 255

    #alpha (transparency)
    alpha = render.transparency

    b /= 255
    g /= 255
    r /= 255
    

    b = (b * alpha) + (b_background * (1- alpha))
    g = (g * alpha) + (g_background * (1- alpha))
    r = (r * alpha) + (r_background * (1- alpha))

    return r,g,b
#para los tres canales de color r,g,b
# colorFinal = (colorCono * transparencia) + (colorAtras * (1 - transparencia)) 


def zBufferFade(render, **kwargs):

    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']
    pixel = kwargs['pixel']

    xyZbuffer = pixel.z

    b /= 255
    g /= 255
    r /= 255

    if render.active_texture:
        tx = ta.x * u + tb.x * v + tc.x * w
        ty = ta.y * u + tb.y * v + tc.y * w
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255
    
    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w 
    nz = na[2] * u + nb[2] * v + nc[2] * w 

    normal = V3(nx, ny, nz)

    intesity = dot(normal, render.light)

    b *=  xyZbuffer
    g *=  xyZbuffer
    r *=  xyZbuffer

    if xyZbuffer >= 0:
        return r,g,b
    else:
        return 0,0,0

def normalMap(render, **kwargs):
    A, B, C = kwargs['verts']
    u, v, w = kwargs['baryCoords']
    ta, tb, tc = kwargs['texCoords']
    na, nb, nc = kwargs['normals']
    b, g, r = kwargs['color']

    b /= 255
    g /= 255
    r /= 255

    tx = ta.x * u + tb.x * v + tc.x * w
    ty = ta.y * u + tb.y * v + tc.y * w

    if render.active_texture:
        texColor = render.active_texture.getColor(tx, ty)
        b *= texColor[0] / 255
        g *= texColor[1] / 255
        r *= texColor[2] / 255

    nx = na[0] * u + nb[0] * v + nc[0] * w
    ny = na[1] * u + nb[1] * v + nc[1] * w
    nz = na[2] * u + nb[2] * v + nc[2] * w
    normal = V3(nx, ny, nz)

    if render.active_normalMap:
        texNormal = render.active_normalMap.getColor(tx, ty)
        texNormal = V3((texNormal[2] / 255) * 2 - 1,
                      (texNormal[1] / 255) * 2 - 1,
                      (texNormal[0] / 255) * 2 - 1)

        texNormal = norm(texNormal)

        edge1 = sub(B,A)
        edge2 = sub(C,A)
        deltaUV1 = subV2(tb, ta)
        deltaUV2 = subV2(tc, ta)
        tangent = [0,0,0]
        f = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
        tangent[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0])
        tangent[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1])
        tangent[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
        tangent = V3(tangent[0], tangent[1], tangent[2])
        tangent = sub(tangent, mul(normal, dot(tangent, normal)))
        tangent = norm(tangent)

        bitangent = cross(normal, tangent)
        bitangent = norm(bitangent)

        #para convertir de espacio global a espacio tangente
        tangentMatrix = [[tangent[0],bitangent[0],normal[0]],
                                [tangent[1],bitangent[1],normal[1]],
                                [tangent[2],bitangent[2],normal[2]]]

        light = render.light
        light = matXvect(tangentMatrix, light)
        light = V3(light[0][0],light[0][1], light[0][2])
        light = norm(light)
        intensity = dot(texNormal, light) #v3
    else:
        intensity = dot(normal, render.light)

    b *= intensity
    g *= intensity
    r *= intensity

    if intensity > 0:
        return r, g, b
    else:
        return 0,0,0
        
