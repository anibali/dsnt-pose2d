from PIL.ImageDraw import Draw

# Joints to connect for visualisation, giving the effect of drawing a
# basic "skeleton" of the pose.
bones = {
  'right_lower_leg':    (0, 1),
  'right_upper_leg':    (1, 2),
  'right_pelvis':       (2, 6),
  'left_lower_leg':     (4, 5),
  'left_upper_leg':     (3, 4),
  'left_pelvis':        (3, 6),
  'center_lower_torso': (6, 7),
  'center_upper_torso': (7, 8),
  'center_head':        (8, 9),
  'right_lower_arm':    (10, 11),
  'right_upper_arm':    (11, 12),
  'right_shoulder':     (12, 8),
  'left_lower_arm':     (14, 15),
  'left_upper_arm':     (13, 14),
  'left_shoulder':      (13, 8),
}

def draw_skeleton(img, coords, joint_mask=None):
  """Draw a pose skeleton connecting joints (for visualisation purposes).

  Left-hand-side joints are connected with blue lines. Right-hand-size joints
  are connected with red lines. Center joints are connected with magenta
  lines.

  Args:
    img (PIL.Image.Image): PIL image which the skeleton will be drawn over.
    coords (Tensor): 16x2 tensor containing 0-based pixel coordinates
      of joint locations. Joints indices are expected to match
      http://human-pose.mpi-inf.mpg.de/#download
    joint_mask (Tensor, optional): Mask of valid joints (invalid joints
      will be drawn with grey lines).
  """

  draw = Draw(img)
  for bone_name, (j1, j2) in bones.items():
    if bone_name.startswith('center_'):
      colour = (255, 0, 255)  # Magenta
    elif bone_name.startswith('left_'):
      colour = (0, 0, 255)    # Blue
    elif bone_name.startswith('right_'):
      colour = (255, 0, 0)    # Red
    else:
      colour = (255, 255, 255)

    if joint_mask is not None:
      # Change colour to grey if either vertex is not masked in
      if joint_mask[j1] == 0 or joint_mask[j2] == 0:
        colour = (100, 100, 100)

    draw.line([coords[j1, 0], coords[j1, 1], coords[j2, 0], coords[j2, 1]], fill=colour)
