import numpy as np

def unit(v):
    """normalize a list of vectors v
    """

    return v/np.linalg.norm(v, axis=1)[:, None]

    



def circle_3pts(p1, p2, p3):
    """
    Function that take three list of points
    return the center and radius of the circle that pass through them.
    """ 

    # Check if p1, p2, p3 are list of points or single points
    if len(p1.shape) == 1:
        p1 = p1[None, :]
    if len(p2.shape) == 1:
        p2 = p2[None, :]
    if len(p3.shape) == 1:
        p3 = p3[None, :]
    

    # Create local coordinate system
    
    u1 = unit(p2-p1) 
    u2 = unit( np.cross(p3-p1, u1) ) # axis direction
    u3 = np.cross(u2, u1)
    
    # Find the center and radius in the new system

    # bx = p2-p1 . u1
    bx = np.sum((p2 - p1)*u1, axis=1)
    cx = np.sum((p3 - p1)*u1, axis=1)
    cy = np.sum((p3 - p1)*u3, axis=1)


    # h 
    # h = ( (cx - bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    h = ((cx-bx/2)**2 + cy**2 - (bx/2)**2 )/(2*cy)
    
    bx /= 2

    radius = np.linalg.norm(np.vstack((bx,h)),axis=0)


    center = p1 + (bx)[:,None]*u1 + h[:,None]*u3 
    
    return center, radius, u2