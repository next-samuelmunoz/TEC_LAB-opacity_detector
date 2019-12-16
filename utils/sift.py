# -*- coding: utf-8 -*-


from collections import defaultdict
import itertools

import cv2
import numpy as np
from scipy.spatial import distance
from scipy.spatial import KDTree

'''
Similarity functions
f: function(v1, v2)  # v is a vector
t: maximum threshold
'''
SIMILARITIES={
    'euclidean':{
        'f': distance.euclidean,
        't': 250
    },
    'cosine':{
        'f': distance.cosine,
        't': 0.2
    },
    'braycurtis':{
        'f': distance.braycurtis,
        't': 0.3
    },
    'chebishev':{
        'f': distance.chebyshev,
        't': 100
    },
    'sqeuclidean':{
        'f': distance.sqeuclidean,
        't': 90000
    }
}


def extract_kps(img):
    """Extract SIFT keypoints

    PARAMETERS:
    img: opnecv image
        Image to calculate keypoints.

    RETURN:
    retval: dict {(x,y): {'kp':[], 'desc':[]}}
        Dictionary (spatial position) of keypoints.
    """
    sift = cv2.xfeatures2d.SIFT_create()
    return {
        kp.pt: {
            'kp': kp,
            'desc': desc
        }
        for kp,desc in
        zip(*sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None))
    }


def pair_similarity(kp1, kp2, sim_func,
    dist_func=distance.euclidean,
    dist_factor=0.5
):
    """Calculate the similarity between 2 points

    PARAMETERS:
    kp1: keypoint 1
    kp2: keypoint 2
    sim_func: similarity function for sift descriptor
    dist_func: spatial function (euclidean) for x,y
    dist_factor: weight for spatial component VS descriptor component

    RETURN:
    retval: float
    """
    return (
        dist_factor * dist_func(kp1['kp'].pt, kp2['kp'].pt) +
        (1-dist_factor) * sim_func(kp1['desc'], kp2['desc'])
    )


def _calculate_pair_points(kps1, kps2, dist_threshold, sim_func, sim_threshold, kdtree1=None, kdtree2=None):
    """Calculate similar points

    NOTE: kdtrees should be calculated only once.
    """
    retval = []

    # Filter by spatial proximity
    if not kdtree1:
        kdtree1 = KDTree([x['kp'].pt for x in kps1])
    if not kdtree2:
        kdtree2 = KDTree([x['kp'].pt for x in kps2])
    spatial_cands = kdtree1.query_ball_tree(kdtree2, dist_threshold)  # List of list of indices per point

    # Filter by similarity
    for i1,i2s in enumerate(spatial_cands):
        kp1 = kps1[i1]
        kp2_cands = [ # Filter by threshold
            kp2
            for dsimilarity, kp2 in
            [ # Calculate similarity with kp1
                (
                    sim_func(kp1['desc'], kp2['desc']),
                    kp2
                )
                for kp2 in
                [ kps2[i2] for i2 in i2s ]  # Get Keypoints kp2
            ]
            if dsimilarity<=sim_threshold
        ]
        if kp2_cands:
            retval.append((kp1, kp2_cands))
    return retval


def calculate_opacities(imgs_kps, dist_threshold, sim_func, sim_threshold):
    '''Calculate opacity points between different-perspective images
    of the same scene.

    RETURN:
    retval: [img_pair][kp from img1]->(kp img1, [similar kps from img2])
    '''
    retval = []
    for img1_kps, img2_kps in itertools.combinations(imgs_kps, 2):
        opacities_kps = _calculate_pair_points(
            kps1=list(img1_kps.values()),
            kps2=list(img2_kps.values()),
            dist_threshold=dist_threshold,
            sim_func=sim_func,
            sim_threshold=sim_threshold
        )
        retval.append(opacities_kps)
        # print("IMG1 points: {}".format(len(opacities_kps)))
        # print("IMG2 points: {} (with duplicates)".format(sum([len(x) for x in opacities_kps])))
        # imshow_opacities(img1, img2, opacities_kps)
    return retval


def opacities2confidence(opacities_sift, sim_func):
    # Point: [scores of similar points]
    point_scores = defaultdict(lambda : [])  # (x,y): [scores] for all points

    for kps_pair in opacities_sift:
        for kp1, kps2 in kps_pair:
            for kp2 in kps2:
                score = sim_func(kp1, kp2)
                point_scores[kp1['kp'].pt].append(score)
                point_scores[kp2['kp'].pt].append(score)
    # Confidence per point
    # Create pairs (point, avg score)
    point_avg = [
        (point, 1/np.average(scores)) # Similar points have bigger score now
        for point, scores in point_scores.items()
    ]
    # Normalize 0-1, scale factor (biggest score = 1)
    scores = [score for _,score in point_avg]
    #s_min = min(scores)
    #s_range = max(scores) - s_min
    s_max = max(scores)
    point_confidence = sorted(
        [
            #(point, (score-s_min)/s_range)
            (point, (score/s_max))
            for point, score in point_avg
        ],
        key=lambda x:x[1],
        reverse=True
    )
    return point_confidence
