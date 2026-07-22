# Phase 6.1 step 0 - hand-pose measurements

REST = (-0.213, -0.09, 0.2)   GRAB = (-0.01, 0.006, 0.33)   (left-hand agent-local)

## M1 - Does REST keep the hand out of the camera frame?
- at a SHELF:  
- in an AISLE:  
- verdict (REST usable? re-pick pose?):  

## M2 - Does REST survive LiDAR (clearance gate)?
- clearance @ REST vs hands-off:  
- verdict (self-culled?):  

## M3 - Does a gripped item survive moves + a full checkpoint drive?
- route driven:  
- drops:  
- verdict:  

## M4 - Does a carried item occlude the camera / LiDAR centre ray?
- centre-ray dist empty vs holding:  
- item in frame?:  
- verdict (known cost?):  
