
 
# Pt3O4 strcutures
################################################################### 
class Pt3O4Lattice(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):
       
        self.init_containers()
 
        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)
 
        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]
       
        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
       self.beta = radians(90)
        self.gamma = radians(90)
       
        self.sigma_D = sigma_D          # Lattice disorder
       
        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'hexoctahedral'
        self.symmetry['point group'] = 'm3m'
        self.symmetry['space group'] = 'Cp14,223'               
        
        self.positions = ['faces', 'edges', 'tetra']
      
        self.lattice_positions = ['faceXY','faceYZ','faceZX','edgeX','edgeY','edgeZ','tetra1', 'tetra2','tetra3','tetra4','tetra5','tetra6','tetra7','tetra8']
 
        
        self.lattice_coordinates = [ (0.5, 0.5, 0.0), \
                                        (0.0, 0.5, 0.5), \
                                        (0.5, 0.0, 0.5), \
                                        (0.5, 0.0, 0.0), \
                                        (0.0, 0.5, 0.0), \
                                        (0.0, 0.0, 0.5), \
 
                                        (0.25, 0.25, 0.25), \
                                        (0.25, 0.25, 0.75), \
                                        (0.25, 0.75, 0.25), \
                                        (0.75, 0.25, 0.25), \
                                        (0.25, 0.75, 0.75), \
                                        (0.75, 0.25, 0.75), \
                                        (0.75, 0.75, 0.25), \
                                        (0.75, 0.75, 0.75), \
                                    ]
        self.lattice_objects = [ self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
                                    self.objects[0], \
 
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                    self.objects[1], \
                                   
                                ]
                               
 
 
 
    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
       
        return 1
 
 
 
    def unit_cell_volume(self):
       
        return self.lattice_spacing_a**3
 
 
 
 
 
# Cu5Zn8 strcutures
################################################################### 
class Cu5Zn8Lattice(Lattice):
    def __init__(self, objects, lattice_spacing_a=1.0, sigma_D=0.01):
       
        self.init_containers()
 
        self.min_objects = 1
        self.expected_objects = 2
        self.object_count(objects)
 
        if len(objects)==1:
            # Assume same object everywhere
            self.objects = [ objects[0], objects[0] ]
        else:
            # We only need four objects. Ignore everything else.
            self.objects = objects[0:self.expected_objects]
       
        self.lattice_spacing_a = lattice_spacing_a
        self.lattice_spacing_b = lattice_spacing_a
        self.lattice_spacing_c = lattice_spacing_a
        self.alpha = radians(90)
        self.beta = radians(90)
        self.gamma = radians(90)
       
        self.sigma_D = sigma_D          # Lattice disorder
       
        # Define the lattice
        self.symmetry = {}
        self.symmetry['crystal family'] = 'cubic'
        self.symmetry['crystal system'] = 'cubic'
        self.symmetry['Bravais lattice'] = 'P'
        self.symmetry['crystal class'] = 'XX'
        self.symmetry['point group'] = 'XXX'
        self.symmetry['space group'] = 'cI52,217'               
        
        self.positions = ['II']
        self.lattice_positions = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20', '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',  '41','42','43','44','45','46','47','48','49','50','51','52']
       
 
 
       
        self.lattice_coordinates = [ (0.1872, 0.1872, 0.5366), \
                                        (0.6442,0,0),   \
                                        ( 0.6089 ,      0.6089 ,      0.6089 ),     \
                                        ( 0.828  ,      0.828  ,      0.828  ),     \
                                        ( 0.8128 ,      0.8128 ,      0.5366 ),     \
                                        ( 0.8128 ,      0.1872 ,      0.4634 ),     \
                                        ( 0.1872 ,      0.8128 ,      0.4634 ),     \
                                        ( 0.5366 ,      0.1872 ,      0.1872 ),     \
                                        ( 0.1872 ,      0.5366 ,      0.1872 ),     \
                                        ( 0.5366 ,      0.8128 ,      0.8128 ),     \
                                        ( 0.8128 ,      0.5366 ,      0.8128 ),     \
                                        ( 0.4634 ,      0.8128 ,      0.1872 ),     \
                                        ( 0.1872 ,      0.4634 ,      0.8128),      \
                                        ( 0.4634 ,      0.1872 ,      0.8128 ),     \
                                        ( 0.8128 ,      0.4634 ,      0.1872 ),     \
                                        ( 0.3558 ,      0      ,      0      ),     \
                                        ( 0      ,      0.6442 ,      0      ),     \
                                        ( 0      ,      0      ,      0.6442 ),     \
                                        ( 0      ,      0.3558 ,      0      ),     \
                                        ( 0      ,      0      ,      0.3558 ),     \
                                        ( 0.3911 ,      0.3911 ,      0.6089 ),     \
                                        ( 0.3911 ,      0.6089 ,      0.3911 ),     \
                                        ( 0.6089 ,      0.3911 ,      0.3911 ),     \
                                        ( 0.172  ,      0.172  ,      0.828  ),     \
                                        ( 0.172  ,      0.828  ,      0.172  ),     \
                                        ( 0.828  ,      0.172  ,      0.172  ),     \
                                        ( 0.6872 ,      0.6872 ,      0.0366 ),     \
                                        ( 0.1442 ,      0.5    ,      0.5    ),     \
                                        ( 0.1089 ,      0.1089 ,      0.1089 ),     \
                                        ( 0.328  ,      0.328  ,      0.328  ),     \
                                        ( 0.3128 ,      0.3128 ,      0.0366 ),     \
                                        ( 0.3128 ,      0.6872 ,      0.9634 ),     \
                                        ( 0.6872 ,      0.3128 ,      0.9634 ),     \
                                        ( 0.0366 ,      0.6872 ,      0.6872 ),     \
                                        ( 0.6872 ,      0.0366 ,      0.6872 ),     \
                                        ( 0.0366 ,      0.3128 ,      0.3128 ),     \
                                        ( 0.3128 ,      0.0366 ,      0.3128 ),     \
                                        ( 0.9634 ,      0.3128 ,      0.6872 ),     \
                                        ( 0.6872 ,      0.9634 ,      0.3128 ),     \
                                        ( 0.9634 ,      0.6872 ,      0.3128 ),     \
                                        ( 0.3128 ,      0.9634 ,      0.6872 ),     \
                                        ( 0.8558 ,      0.5    ,      0.5    ),     \
                                        ( 0.5    ,      0.1442 ,      0.5    ),     \
                                        ( 0.5    ,      0.5    ,      0.1442 ),     \
                                        ( 0.5    ,      0.8558 ,      0.5    ),     \
                                        ( 0.5    ,      0.5    ,      0.8558 ),     \
                                        ( 0.8911 ,      0.8911 ,      0.1089 ),     \
                                        ( 0.8911 ,      0.1089 ,      0.8911 ),     \
                                        ( 0.1089 ,      0.8911 ,      0.8911 ),     \
                                        ( 0.672  ,      0.672  ,      0.328  ),     \
                                        ( 0.672  ,      0.328  ,      0.672  ),     \
                                        ( 0.328  ,      0.672  ,      0.672 ),      \
 
 
 
 
                                    ]
 
 
 
 
 
 
        self.lattice_objects = [ self.objects[1], \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[1]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
                                    self.objects[0]     ,      \
 
 
 
 
                                   
                                ]
                               
 
 
 
    def symmetry_factor(self, h, k, l):
        """Returns the symmetry factor (0 for forbidden)."""
       
        return 1
 
 
 
    def unit_cell_volume(self):
       
        return self.lattice_spacing_a**3
 
  
