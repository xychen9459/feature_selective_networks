import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' test case. in this case, batch size would be 2.
testroi = RoIAlign_withBanks().cuda()
top_absolute_net_locs = torch.tensor([[0.1,0.1,0.3,0.4],[0.,0,0.5,0.5]]).float().cuda()
which_image_ind = np.array([1,0])
feature_maps = [torch.rand((2,512,x,x)).float().cuda() for x in [38,19,10,5,3,1]]
testout = testroi.forward(top_absolute_net_locs, which_image_ind, feature_maps)
'''

class RoIAlign_withBanks(nn.Module):
    """
    https://arxiv.org/abs/1711.08879
    This generates aspect ratio and sub-region banks, concats them together, and pools them somehow with the original roi pool
    """
    def __init__(self, output_size = 7, in_channels = 512, out_channels = 40):
        super(RoIAlign_withBanks, self).__init__()
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.default_pool_reduction_conv = nn.Sequential(
            CoordConv(in_channels = in_channels, out_channels = out_channels, stride = 1, kernel_size = 1, padding = 0, bias = False),
            GroupNorm(out_channels),
            nn.RReLU(inplace = True),
        )
            #do you bn/relu here? I'm not sure, so how about yes by default.
        self.aspect_ratios = [0.75, 1.3] #three cases: x < 0.75, 0.75 <= x <= 1.3, 1.3 < x
            #this is width to height? let's say it's width to height.
        self.aspect_ratio_convs = nn.Sequential(*[
            nn.Sequential(
                CoordConv(in_channels = in_channels, out_channels = out_channels, stride = 1, kernel_size = 1, padding = 0, bias = False),
                GroupNorm(out_channels),
                nn.RReLU(inplace = True),
            )
            for i in range(len(self.aspect_ratios)+1)
            ])
        #using 3x3 shift grid
        self.offsets_list = [(-1 + (i%3), -1 + (i//3)) for i in range(9)]
        self.shifted_convs = nn.Sequential(*[
            nn.Sequential(
                ShiftedConv(in_channels = in_channels, out_channels = out_channels, offset_x = thisoffset[0], offset_y = thisoffset[1], kernel_size = 1, stride = 1, padding = 0, bias = False),
                GroupNorm(out_channels),
                nn.RReLU(inplace = True),
            )
            for thisoffset in self.offsets_list
            ])
        self.final_reduction_convs = nn.Sequential(
            CoordConv(in_channels = out_channels*3, out_channels = out_channels, stride = 1, kernel_size = 1, padding = 0, bias = False),
            GroupNorm(out_channels),
            nn.RReLU(inplace = True),
        )
    def forward(self, top_absolute_net_locs, which_image_ind, feature_maps, use_cuda = True):
        """
        forward the entire bbox coords tensor
        
        Arguments
        top_absolute_net_locs: a Nx4 list of bbox proposals, in the form x1 y1 x2 y2
        which_image_ind: a list of indices to indicate which example each proposal comes from
        feature_maps: a list of feature maps. In my case, I was using a list of six maps with dimensions 38, 19, 10, 5, 3, 1 (in that order), each with shape batch_size x 512 (channels) x dim x dim.
        
        The output is a tensor the same length as the input proposals, with the specified output dim and number of output channels. It's made from pooling from a specified feature map
        """
        #get interpolations
        bbox_gridinterpolations = vectorizedTorchBilinearInterpolationGrid(top_absolute_net_locs, self.output_size, self.output_size, use_cuda)
        #get corresponding fmaps
        map_indices = chooseFeatureMapIndices_fromBboxCoordList_x1y1x2y2(top_absolute_net_locs, len(feature_maps))
        maxfmapsize = feature_maps[min(map_indices)].shape[-1]
        upscaled_feature_maps_tensor = torch.cat([F.upsample(input = feature_maps[i], size = (maxfmapsize, maxfmapsize), mode="bilinear", align_corners = False).unsqueeze(1) for i in range(len(feature_maps))], dim=1)
        #
        squeeze01shape = (upscaled_feature_maps_tensor.shape[0]*upscaled_feature_maps_tensor.shape[1], upscaled_feature_maps_tensor.shape[2], upscaled_feature_maps_tensor.shape[3], upscaled_feature_maps_tensor.shape[4])
        outshape = (len(self.aspect_ratios)+1, upscaled_feature_maps_tensor.shape[0], upscaled_feature_maps_tensor.shape[1], self.out_channels, upscaled_feature_maps_tensor.shape[3], upscaled_feature_maps_tensor.shape[4])
        ratio_banks = torch.cat([self.aspect_ratio_convs[i](upscaled_feature_maps_tensor.view(squeeze01shape)) for i in range(len(self.aspect_ratio_convs))]).view(outshape)
        ###sub-region banks
        sr_outshape = (len(self.offsets_list), upscaled_feature_maps_tensor.shape[0], upscaled_feature_maps_tensor.shape[1], self.out_channels, upscaled_feature_maps_tensor.shape[3], upscaled_feature_maps_tensor.shape[4])
        subregion_banks = torch.cat([self.shifted_convs[i](upscaled_feature_maps_tensor.view(squeeze01shape)) for i in range(len(self.shifted_convs))]).view(sr_outshape)
        #
        reduced_default_feature_maps_tensor = self.default_pool_reduction_conv(upscaled_feature_maps_tensor.view(squeeze01shape)).view((upscaled_feature_maps_tensor.shape[0], upscaled_feature_maps_tensor.shape[1], self.out_channels, maxfmapsize, maxfmapsize))
        selected_fmaps_tensor_default = torch.cat([reduced_default_feature_maps_tensor[which_image_ind[i], map_indices[i]].unsqueeze(0) for i in range(len(map_indices))], dim=0)
        pooled_regions_default = getWeightedSum_ofFourFmapPoints_forFmapTensorAndBboxGridTensor(selected_fmaps_tensor_default, bbox_gridinterpolations, use_cuda)
        #####now selectively pool the aspect ratio and subregion rois from their tensors
        netloc_dims = (top_absolute_net_locs[:,2:] - top_absolute_net_locs[:,0:2]).detach().cpu().numpy()
        netloc_ratios_widthtoheight = np.where(netloc_dims[:,1] <= 0., netloc_dims[:,0]*0+1, netloc_dims[:,0]/netloc_dims[:,1])
        which_aspectratio_ind = np.where(netloc_ratios_widthtoheight < self.aspect_ratios[0], 0, np.where(netloc_ratios_widthtoheight > self.aspect_ratios[1], 2, 1))
            #ratio_banks #torch.Size([3, 2, 6, 40, 38, 38])
        selected_fmaps_tensor_aspectratio = torch.cat([ratio_banks[which_aspectratio_ind[i], which_image_ind[i], map_indices[i]].unsqueeze(0) for i in range(len(map_indices))], dim=0)
        pooled_regions_aspectratio = getWeightedSum_ofFourFmapPoints_forFmapTensorAndBboxGridTensor(selected_fmaps_tensor_aspectratio, bbox_gridinterpolations, use_cuda)
        ##pool again for subregions
        netloc_centers_xy = torch.cat([(top_absolute_net_locs[:,0:1] + top_absolute_net_locs[:,2:3])/2, (top_absolute_net_locs[:,1:2] + top_absolute_net_locs[:,3:4])/2], dim=1).detach().cpu().numpy()
            #in my case, the net_locs are width then height
        which_x_ind = np.where(netloc_centers_xy[:,0] > 2/3., 2, np.where(netloc_centers_xy[:,0] < 1/3., 0, 1))
        which_y_ind = np.where(netloc_centers_xy[:,1] > 2/3., 2, np.where(netloc_centers_xy[:,1] < 1/3., 0, 1))
        which_subregion_ind = 3*which_y_ind + which_x_ind
            #subregion_banks #torch.Size([9, 2, 6, 40, 38, 38])
        selected_fmaps_tensor_subregion = torch.cat([subregion_banks[which_subregion_ind[i], which_image_ind[i], map_indices[i]].unsqueeze(0) for i in range(len(map_indices))], dim=0)
        pooled_regions_subregion = getWeightedSum_ofFourFmapPoints_forFmapTensorAndBboxGridTensor(selected_fmaps_tensor_subregion, bbox_gridinterpolations, use_cuda)
        ####now combine the subregion and aspect ratio maps
            #paper uses elementwise sum on the aspect ratio and subregion banks.
            #but then wtf does it do to the default bank and the summed bank to get a bank the same size? It uses a different symbol in the diagram.
            #So you know what? I'm just going to concatenate all three maps because I know it works.
        concatted_attention_maps = torch.cat((pooled_regions_default, pooled_regions_aspectratio, pooled_regions_subregion), dim=1)
        #then, since my model wants the output shape, let's just have a conv reduce the number of features
        slim_output_maps = self.final_reduction_convs(concatted_attention_maps)
        return slim_output_maps
class ShiftedConv(nn.Module):
    def __init__(self, in_channels, out_channels, offset_x, offset_y, kernel_size = 1, stride = 1, padding = 0, bias = False, groups = 1, dilation = (1,1)):
        '''
        shifts feature map offset_x right and offset_y down before conv-ing.
        for now, just assume integer offsets.
        [][][]but it's really not that hard to do a bilinearly interpolated shift.
        '''
        super(ShiftedConv, self).__init__()
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.mainconv = CoordConv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, groups = groups, dilation = dilation)
    def forward(self,x):
        x = tensor_roll(x, self.offset_x, axis=-2)
        x = tensor_roll(x, self.offset_y, axis=-1)
        out = self.mainconv(x)
        return out
def tensor_roll(tensor, shift, axis, wraparound = False, mode = "constant"):
    '''
    roll tensor. modified from
    https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/5
    in deformed conv, this shouldn't wrap around; it should either pad with blank or repeat/reflect the exterior. I'll just repeat ("replicate") here. or maybe constant is better. hm, I'll use constant.
    '''
    shift_init = shift
    if shift == 0:
        return tensor
    if axis < 0:
        axis += tensor.dim()
    if not wraparound: #pad the tensor by shift along the specified dim
        padding = getTensorPadding(tensor, shift, axis)
        tensor = F.pad(tensor, padding, mode=mode)
    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)
    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    if wraparound:
        return torch.cat([after, before], axis)
    elif not wraparound:
        if shift_init > 0:
            return before
        else:
            return after
def getTensorPadding(tensor, shift, axis):
    if shift == 0:
        return (0,0,0,0)
    if axis == tensor.dim()-1 or axis == -1:
        if shift > 0:
            return (abs(shift),0,0,0)
        else:
            return (0,abs(shift),0,0)
    elif axis == tensor.dim()-2 or axis == -2:
        if shift > 0:
            return (0,0,abs(shift),0)
        else:
            return (0,0,0,abs(shift))
    else:
        print("can't pad; axis should be -1 or -2")
        return None
    return padding
#
def vectorizedTorchBilinearInterpolationGrid(bboxcoordslist, grid_width, grid_height, use_cuda):
    """
    transform a tensor list of bbox point ranges from x1 to x2 (and y1 to y2) into a WxH bilinear interpolated grid.
    input: Nx4
    output: N x grid_width x grid_height x 2. The 2 is for x, y coordinates.
    """
    if grid_width == 0 or grid_height == 0:
        print("grid width or grid height is 0. returning")
        return None
    x1 = bboxcoordslist[:, 0:1]
    x2 = bboxcoordslist[:, 2:3]
    y1 = bboxcoordslist[:, 1:2]
    y2 = bboxcoordslist[:, 3:4]
    if use_cuda:
        x_ranges = torch.arange(grid_width).cuda()*(x2-x1)/max(1.,(grid_width-1.)) + x1
        y_ranges = torch.arange(grid_height).cuda()*(y2-y1)/max(1.,(grid_height-1.)) + y1
    else:
        x_ranges = torch.arange(grid_width)*(x2-x1)/max(1.,(grid_width-1.)) + x1
        y_ranges = torch.arange(grid_height)*(y2-y1)/max(1.,(grid_height-1.)) + y1
    #Now for each row in these range lists--that's two Nx7 tensors, where each row is the x1 to x2 or y1 to y2 range corresponding to an example--I want to make a grid out of the x and y values. Numpy has meshgrid. Does torch have something similar?
    #return np.array(np.meshgrid(x_range, y_range)).transpose(2,1,0)
    bbox_interpolations = torch_combined_meshgrid(x_ranges, y_ranges)
    bbox_interpolations = bbox_interpolations.clamp(0,1)
    return bbox_interpolations
def torch_combined_meshgrid(x_ranges, y_ranges):
    """
    convert a NxW x_ranges and a NxH y_ranges into a NxWxHx2 list of coordinate grids.
    To be fed into a mapping function to turn each point into a weighted average of the four nearest points from a feature map associated with each example.
    """
    grid_width = x_ranges.shape[1]
    grid_height = y_ranges.shape[1]
    x_grid = x_ranges.repeat(1, grid_height).view(x_ranges.shape[0], grid_width, grid_height, 1)
    y_grid = y_ranges.repeat(1, grid_width).view(x_ranges.shape[0], grid_width, grid_height, 1).transpose(1,2)
    output = torch.cat((x_grid, y_grid), dim=-1)
    output = output.transpose(2,1)
    #transpose is so output[0,2,1] will get 3rd x and 2nd y coordinate, rather than [0,1,2] being used to get that.
    return output
#single item stuff
def numpyBilinearInterpolationGrid(x1, y1, x2, y2, grid_width, grid_height):
    """
    transform a bbox point range from x1 to x2 (and y1 to y2) into a WxH bilinear interpolated grid.
    Uses numpy.
    """
    if grid_width == 0 or grid_height == 0:
        print("grid width or grid height is 0. returning")
        return None
    x_range = np.arange(grid_width)*(x2-x1)/max(1.,(grid_width-1.)) + x1
    y_range = np.arange(grid_height)*(y2-y1)/max(1.,(grid_height-1.)) + y1
    return np.array(np.meshgrid(x_range, y_range)).transpose(2,1,0)
def chooseFeatureMapIndices_fromBboxCoordList_x1y1x2y2(bboxcoordslist, len_feature_maps = 6):
    """
    bboxcoordslist: Nx4 x1 y1 x2 y2 bbox coords tensor. Coordinates are normed from 0 to 1.
    len_feature_maps: length of feature maps list. My feature maps has a 38x38 feature map first, then the last map is 1x1, so the largest index is used to select the 1x1 map, which is appropriate for a full-size bbox which covers the entire image.
    #https://arxiv.org/abs/1612.03144 Feature Pyramid Networks for Object Detection. Page 4 shows a way to select which level to use. floor(k_0 + log_2(sqrt(W*H)/im_size))
    #in my case, I would use max(0, floor(5 + log_2(sqrt(W*H)/im_size))) to choose the index; I have 6 feature maps.
    """
    x1, y1, x2, y2 = bboxcoordslist.split(1, dim=1)
    widths = x2 - x1
    heights = y2 - y1
    avg_dims = torch.sqrt(widths*heights).cpu()
    log2_dims = torch.log2(avg_dims)
    map_indices = torch.tensor(len_feature_maps - 1.) + log2_dims
    map_indices = torch.floor(map_indices)
    map_indices = torch.clamp(map_indices, 0, len_feature_maps-1)
    map_indices = map_indices.int().view(-1).detach().numpy()
    return map_indices
def getWeightedSum_ofFourFmapPoints_forFmapTensorAndBboxGridTensor(selected_fmaps_tensor, bbox_gridinterpolations, use_cuda = True):
    """
    transform a point inside a feature map into a weighted average of the four nearest features, based on the distance of the point from each feature point. I think that's bilinear interpretation?
    This is used in RoIAlign
    """
    fmap_dim = selected_fmaps_tensor.shape[-1]
    if fmap_dim == 1:
        return selected_fmaps_tensor[:, :, 0:1, 0:1]
    #fmap_point_start = np.floor(bbox_point*(fmap_dim-1))/(fmap_dim-1) #have to use dim-1 here because the arrays go from 0 to n-1
    fmap_point_start = torch.floor(bbox_gridinterpolations*(fmap_dim-1))/(fmap_dim-1)
    fmap_point_end = torch.ceil(bbox_gridinterpolations*(fmap_dim-1))/(fmap_dim-1)
    distance_from_start = bbox_gridinterpolations - fmap_point_start
    #distance_from_end = fmap_point_end - bbox_gridinterpolations
    proportion_to_take_from_start_x = torch.ones(bbox_gridinterpolations.shape[0:3])
    proportion_to_take_from_start_y = torch.ones(bbox_gridinterpolations.shape[0:3])
    if use_cuda:
        proportion_to_take_from_start_x = proportion_to_take_from_start_x.cuda()
        proportion_to_take_from_start_y = proportion_to_take_from_start_y.cuda()
    difference_x = (fmap_point_end[:,:,:,0] - fmap_point_start[:,:,:,0])
    difference_y = (fmap_point_end[:,:,:,1] - fmap_point_start[:,:,:,1])
    difference_x_is_zero = difference_x == 0
    difference_y_is_zero = difference_y == 0
    difference_x_zeros_set_to_ones = difference_x + difference_x_is_zero.float()
    difference_y_zeros_set_to_ones = difference_y + difference_y_is_zero.float()
        #this is for say, if x1 and x2 are the same value. distance between the two will be 0 and we don't want to divide by 0. In this case, the proportions should be set to use x1*1 and x2*0.
    proportion_to_take_from_start_x = 1. - distance_from_start[:,:,:,0] / difference_x_zeros_set_to_ones
    proportion_to_take_from_end_x = 1. - proportion_to_take_from_start_x
    proportion_to_take_from_start_y = 1. - distance_from_start[:,:,:,1] / difference_y_zeros_set_to_ones
    proportion_to_take_from_end_y = 1. - proportion_to_take_from_start_y
    #now to get the weighted sum...
    fmap_index_start = torch.round(fmap_point_start*(fmap_dim-1)).long()
    fmap_index_end = torch.round(fmap_point_end*(fmap_dim-1)).long()
    #tl = feature_map[0, :, fmap_index_start[0], fmap_index_start[1]]
    #now what should the correct top left output shape be? I think it should be 12x512x7x7
    #tl = selected_fmaps_tensor[:, :, fmap_index_start[:,:,:,0], fmap_index_start[:,:,:,1]]
        #this junk blob returns 12, 512, 12, 7, 7 shape
    tl = torch.cat([selected_fmaps_tensor[i:i+1, :, fmap_index_start[i,:,:,0], fmap_index_start[i,:,:,1]] for i in range(selected_fmaps_tensor.shape[0])], dim=0)
        #that's a bit messy because of the list comprehension, but it seems to work.
        #[][][]how would I do that without the list comprehension?
    tr = torch.cat([selected_fmaps_tensor[i:i+1, :, fmap_index_end[i,:,:,0], fmap_index_start[i,:,:,1]] for i in range(selected_fmaps_tensor.shape[0])], dim=0)
    bl = torch.cat([selected_fmaps_tensor[i:i+1, :, fmap_index_start[i,:,:,0], fmap_index_end[i,:,:,1]] for i in range(selected_fmaps_tensor.shape[0])], dim=0)
    br = torch.cat([selected_fmaps_tensor[i:i+1, :, fmap_index_end[i,:,:,0], fmap_index_end[i,:,:,1]] for i in range(selected_fmaps_tensor.shape[0])], dim=0)
    reshapeshape = (bbox_gridinterpolations.shape[0], 1, bbox_gridinterpolations.shape[1], bbox_gridinterpolations.shape[2])
    p_tl = (proportion_to_take_from_start_x*proportion_to_take_from_start_y).view(reshapeshape)
    p_tr = (proportion_to_take_from_end_x*proportion_to_take_from_start_y).view(reshapeshape)
    p_bl = (proportion_to_take_from_start_x*proportion_to_take_from_end_y).view(reshapeshape)
    p_br = (proportion_to_take_from_end_x*proportion_to_take_from_end_y).view(reshapeshape)
    weighted_sum = p_tl*tl + p_tr*tr + p_bl*bl + p_br*br
    return weighted_sum

########
##coordconv
class CoordConv(nn.Module):
    """
    https://arxiv.org/abs/1807.03247
    convolution, but with appending two x and y coordinate channels before the conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False, groups = 1, dilation = (1,1)):
        super(CoordConv, self).__init__()
        self.theconv = nn.Conv2d(in_channels = in_channels+2, out_channels = out_channels if groups == 1 else in_channels + 2, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias, groups = in_channels+2 if groups > 1 else groups, dilation = dilation)
        self.groups = groups
        if self.groups > 1:
            self.dwiseconv_stripxy = nn.Conv2d(in_channels = in_channels + 2, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
    def forward(self,x):
        iscuda = next(self.parameters()).is_cuda #have to check for cuda when appending the channels.
        x = makeXChannel_andCatIt(x, iscuda)
        x = makeYChannel_andCatIt(x, iscuda)
        out = self.theconv(x)
        if self.groups > 1:
           out = self.dwiseconv_stripxy(out)
        return out
class CoordConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, padding = 1, output_padding = 1, bias = False, groups = 1, dilation = (1,1)):
        super(CoordConvTranspose, self).__init__()
        self.theconv = nn.ConvTranspose2d(in_channels = in_channels+2, out_channels = out_channels if groups == 1 else in_channels + 2, kernel_size = kernel_size, stride = stride, padding = padding, output_padding = output_padding, bias = bias, groups = in_channels+2 if groups > 1 else groups, dilation = dilation)
        self.groups = groups
        if self.groups > 1:
            print("CoordConvTranspose error: unsure how to handle groups > 1. implement.")
        #    self.dwiseconv_stripxy = nn.ConvTranspose2d(in_channels = in_channels + 2, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, output_padding = 0, bias = False)
    def forward(self,x):
        iscuda = next(self.parameters()).is_cuda #have to check for cuda when appending the channels.
        x = makeXChannel_andCatIt(x, iscuda)
        x = makeYChannel_andCatIt(x, iscuda)
        out = self.theconv(x)
        return out
def makeXChannel_andCatIt(mytensor, iscuda):
    dims = mytensor.shape[-2:]
    xdim = dims[0]
    myrange = np.arange(1, step=1/xdim)
    if xdim > 1: #please no divide by 0
        myrange *= xdim/(xdim - 1)
    mychannel = np.repeat(myrange.reshape(xdim, 1), xdim, axis=1).T
    #now I want to append this channel onto each example.
    bs = mytensor.shape[0]
    mystack = np.repeat(mychannel.reshape(1,1,xdim,xdim), bs, axis=0)
    mystack_tensor = torch.tensor(mystack).float()
    if iscuda:
        mystack_tensor = mystack_tensor.cuda()
    out = torch.cat((mytensor, mystack_tensor), dim=1)
    return out
def makeYChannel_andCatIt(mytensor, iscuda):
    dims = mytensor.shape[-2:]
    ydim = dims[1]
    myrange = np.arange(1, step=1/ydim)
    if ydim > 1:
        myrange *= ydim/(ydim - 1)
    mychannel = np.repeat(myrange.reshape(ydim, 1), ydim, axis=1)
    bs = mytensor.shape[0]
    mystack = np.repeat(mychannel.reshape(1,1,ydim,ydim), bs, axis=0)
    mystack_tensor = torch.tensor(mystack).float()
    if iscuda:
        mystack_tensor = mystack_tensor.cuda()
    out = torch.cat((mytensor, mystack_tensor), dim=1)
    return out

############
##GroupNorm
class GroupNorm(nn.Module):
    """
    https://arxiv.org/abs/1803.08494
    Apparently it's better than batchnorm?
    https://github.com/kuangliu/pytorch-groupnorm/blob/master/groupnorm.py
    """
    def __init__(self, num_features, num_groups=None, eps=1e-5):
        super(GroupNorm, self).__init__()
        #self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        ##just changing these to 1D so my previous models can load batchnorm weights without complaining
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        if num_groups is None:
            self.num_groups = seekGroups(num_features)
        else:
            self.num_groups = num_groups
        self.eps = eps
    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0
        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight.view((1,self.num_features,1,1)) + self.bias.view((1,self.num_features,1,1))
def seekGroups(num_channels, divisors = [8,4,3,2]):
    divisible = np.array([num_channels%x == 0 for x in divisors])
    if not divisible.any():
        return num_channels
    else:
        firstvaliddivisor = divisors[np.argwhere(divisible).flatten()[0]]
        return num_channels // firstvaliddivisor
