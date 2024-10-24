#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line RANSAC fitting
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg
import SIFTmatching as sf
import cv2
def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)
def homography_matrix(x1,x2):
    """
    compute homography matrix using ground plane points on image 1 and image 2
    -input:
        x1: floor points on image 1
        x2: floor points on image 2 
    - output 
        Homography matrix
    """
    A= np.ones((2*x1.shape[1],9))
    row=0
    for i in range(x1.shape[1]):
        x_1= x1[0,i]
        y_1= x1[1,i]
        x_2= x2[0,i]
        y_2= x2[1,i]
        line1=[x_1,y_1,1,0,0,0,-x_1*x_2,-x_2*y_1,-x_2]
        line2=[0,0,0,x_1,y_1,1,-x_1*y_2,-y_2*y_1,-y_2]    
        A[row]=np.array(line1)
        row+=1
        A[row]=np.array(line2)
        row+=1
        
    u,S,Vt= np.linalg.svd(A)
    H= Vt[-1]
    H=H.reshape((3,3))
    return H
def compute_descriptors(image_pers_1,image_pers_2):
        sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
        keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
        keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)
        return keypoints_sift_1,descriptors_1,keypoints_sift_2,descriptors_2

def compute_matches(distRatio,minDist,descriptors_1, descriptors_2):
    matchesList = sf.matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = sf.indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    return dMatchesList

def ransac_homography(dMatchesList,inliersSigma=10):
        # parameters of random sample selection
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 4  # number of points needed to compute the fundamental matrix
    thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

    # number m of random samples
    nAttempts =np.round(np.log(1 - P) / np.log(1 - np.power((1 - 0.3), pMinSet)))
    nAttempts = nAttempts.astype(int)
    print('nAttempts = ' + str(nAttempts))

    RANSACThreshold = 3*inliersSigma
    nVotesMax = 0

    rng = np.random.default_rng()
    H_most_voted=[]
    inliers=[]
    for kAttempt in range(nAttempts):

        # sample matches
        xSubSel = rng.choice(dMatchesList, size=pMinSet, replace=False)
        pts1=[]
        pts2=[]
        ind_={}
        for match in xSubSel:
            p1 = keypoints_sift_1[match.queryIdx].pt
            pts1.append([p1[0],p1[1]])
            p2 = keypoints_sift_2[match.trainIdx].pt
            pts2.append([p2[0],p2[1]])
            ind_[match.queryIdx]=match.trainIdx
        pts1=np.array(pts1)
        pts2=np.array(pts2)
        #compute the Homography for these sample points
        H= homography_matrix(pts1,pts2)
        #get points not used for sampling
        other_points_p1=[]
        other_points_p2=[]
        for match in xSubSel:
            dMatchesList.remove(match)
        for match in dMatchesList:
            p1 = keypoints_sift_1[match.queryIdx].pt
            other_points_p1.append([p1[0],p1[1],1.])
            p2 = keypoints_sift_2[match.trainIdx].pt
            other_points_p2.append([p2[0],p2[1]])
            ind_[match.queryIdx]=match.trainIdx
        
                                
        other_points_p1=np.array(other_points_p1)
        other_points_p2=np.array(other_points_p2)
        # Computing the distance from the points to the model
        pts_t=H@other_points_p1.T
        pts_t=pts_t.T
        pts_t[:,0]=pts_t[:,0]/pts_t[:,-1]
        pts_t[:,1]=pts_t[:,1]/pts_t[:,-1]
        pts_t=pts_t[:,:2]

        epsilon= pts_t-other_points_p2
        votes = np.sqrt(np.linalg.norm(epsilon,axis=1)) < RANSACThreshold  #votes
        nVotes = np.sum(votes) # Number of votes

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            votesMax = votes
            H_most_voted = H
    return H_most_voted

def reproj_error(H,matches):
    reproj_error=0.
    inliers=[]
    ct=0
    pts1=[]
    pts2=[]
    for match in matches:
        p1 = keypoints_sift_1[match.queryIdx].pt
        p2 = keypoints_sift_2[match.trainIdx].pt
        p1= [p1[0],p1[1],1.]
        p2=[p2[0],p2[1]]
        pts1.append(p1)
        pts2.append(p2)
        ct+=1
    pts_t=H@np.array(pts1).T
    pts_t=pts_t.T
    pts_t[:,0]=pts_t[:,0]/pts_t[:,-1]
    pts_t[:,1]=pts_t[:,1]/pts_t[:,-1]
    pts_t=pts_t[:,:2]
    pts2=np.array(pts2)
    
    reproj_error= np.sqrt(np.linalg.norm(pts2-pts_t,axis=1)).mean()
    #condition= 
    inliers=[]
    return reproj_error,inliers
              
# if __name__ == '__main__':
#     np.set_printoptions(precision=4,linewidth=1024,suppress=True)

#     # This is the ground truth
#     l_GT = np.array([[2], [1], [-1500]])

#     plt.figure(1)
#     plt.plot([-100, 1800], [0, 0], '--k', linewidth=1)
#     plt.plot([0, 0], [-100, 1800], '--k', linewidth=1)
#     # Draw the line segment p_l_x to  p_l_y
#     drawLine(l_GT, 'g-', 1)
#     plt.draw()
#     plt.axis('equal')

#     print('Click in the image to continue...')
#     plt.waitforbuttonpress()

#     # Generating points lying on the line but adding perpendicular Gaussian noise
#     l_GTNorm = l_GT/np.sqrt(np.sum(l_GT[0:2]**2, axis=0)) #Normalized the line with respect to the normal norm

#     x_l0 = np.vstack((-l_GTNorm[0:2]*l_GTNorm[2],1))  #The closest point of the line to the origin
#     plt.plot([0, x_l0[0]], [0, x_l0[1]], '-r')
#     plt.draw()

#     mu = np.arange(-1000, 1000, 100)
#     inliersSigma = 10 #Standard deviation of inliers
#     xInliersGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu) + np.diag([1, 1, 0]) @ np.random.normal(0, inliersSigma, (3, len(mu)))
#     nInliers = len(mu)

#     # Generating uniformly random points as outliers
#     nOutliers = 5
#     xOutliersGT = np.diag([1, 1, 0]) @ (np.random.rand(3, 5)*3000-1500) + np.array([[0], [0], [1]])

#     plt.plot(xInliersGT[0, :], xInliersGT[1, :], 'rx')
#     plt.plot(xOutliersGT[0, :], xOutliersGT[1, :], 'bo')
#     plt.draw()
#     print('Click in the image to continue...')
#     plt.waitforbuttonpress()

#     x = np.hstack((xInliersGT, xOutliersGT))
#     x = x[:, np.random.permutation(x.shape[1])] # Shuffle the points

#     # parameters of random sample selection
#     spFrac = nOutliers/nInliers  # spurious fraction
#     P = 0.999  # probability of selecting at least one sample without spurious
#     pMinSet = 2  # number of points needed to compute the fundamental matrix
#     thresholdFactor = 1.96  # a point is spurious if abs(r/s)>factor Threshold

#     # number m of random samples
#     nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
#     nAttempts = nAttempts.astype(int)
#     print('nAttempts = ' + str(nAttempts))

#     nElements = x.shape[1]

#     RANSACThreshold = 3*inliersSigma
#     nVotesMax = 0
#     rng = np.random.default_rng()
#     for kAttempt in range(nAttempts):

#         # Compute the minimal set defining your model
#         xSubSel = rng.choice(x.T, size=pMinSet, replace=False)
#         l_model = np.reshape(np.cross(xSubSel[0], xSubSel[1]), (3, 1))

#         normalNorm = np.sqrt(np.sum(l_model[0:2]**2, axis=0))

#         l_model /= normalNorm
#         # Computing the distance from the points to the model
#         res = l_model.T @ x #Since I already have normalized the line with respect the normal the dot product gives the distance

#         votes = np.abs(res) < RANSACThreshold  #votes
#         nVotes = np.sum(votes) # Number of votes

#         if nVotes > nVotesMax:
#             nVotesMax = nVotes
#             votesMax = votes
#             l_mostVoted = l_model


#     drawLine(l_mostVoted, 'b-', 1)
#     plt.draw()
#     plt.waitforbuttonpress()


#     # Filter the outliers and fit the line
#     iVoted = np.squeeze(np.argwhere(np.squeeze(votesMax)))
#     xInliers = x[:, iVoted]

#     plt.plot(xInliers[0, :], xInliers[1, :], 'y*')
#     plt.draw()
#     plt.waitforbuttonpress()

#     # Fit the least squares solution of inliers using svd
#     u, s, vh = np.linalg.svd(xInliers.T)
#     l_ls = vh[-1, :]

#     drawLine(l_ls, 'r--', 1)
#     plt.draw()
#     plt.waitforbuttonpress()

#     # Project the points on the line using SVD
#     s[2] = 0
#     xInliersProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
#     xInliersProjectedOnTheLine /= xInliersProjectedOnTheLine[2, :]

#     plt.plot(xInliersProjectedOnTheLine[0,:], xInliersProjectedOnTheLine[1, :], 'bx')
#     plt.draw()
#     plt.waitforbuttonpress()
#     print('End')

if __name__ == '__main__':
    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    keypoints_sift_1,descriptors_1,keypoints_sift_2,descriptors_2= compute_descriptors(image_pers_1,image_pers_2)
    distRatio=0.8
    minDist=200
    matches=compute_matches(distRatio,minDist,descriptors_1, descriptors_2)
    H= ransac_homography(matches,30)
    err,inliers=reproj_error(H,matches)
    print(err)
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, inliers,
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(err)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()
