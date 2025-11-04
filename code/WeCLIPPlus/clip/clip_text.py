
#For Camel vs Horse
# BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad',
#                         'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign', 'desert', 'field', 'dune',
#                         ]

#For Dog vs Golf Ball
# BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad',
#                         'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign', 'carpet', 'field', 'desert', 'couch', 'bed',
#                         'table', 'chair', 'desk', 'blanket', 'hallway', 'bush', 'branch', 'flower', 'garden', 'park', 'cliff', 'snow', 'ice', 'stream', 'cage',
#                         'fence', 'rooftop',
#                         ]

# BACKGROUND_CATEGORY = [
#     'ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad',
#     'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign','field','desert','chair',
#     'blanket','bush','branch','flower','garden','park','cliff','snow','ice','stream','cage','fence','rooftop',
#     'towel', 'person','human','people','beach','swimming pool','plant','vegetation','lawn','forest','waves',
#     'surfboard','boulder','hill','twig','leaf', 'foliage', 'vine', 'wood'
# ]

BACKGROUND_CATEGORY = [
    # NICO domains
    'autumn', 'dim', 'grass', 'outdoor', 'rock', 'water',

    # Sky / lighting
    'open sky', 'cloudy sky', 'blue sky with clouds', 'twilight sky',
    'low light background', 'dim lighting', 'night scene', 'shadowy background',
    'backlit silhouette', 'indoor low light', 'poorly lit room', 'underexposed scene',
    'neon night street',

    # Roads / urban ground
    'road surface', 'asphalt pavement', 'highway road', 'street intersection',
    'traffic lane markings', 'sidewalk pavement', 'parking lot', 'parking lot lines',
    'rural road',

    # Transport infrastructure
    'railway tracks', 'railway ballast', 'train platform', 'tunnel interior',
    'airport runway', 'airport tarmac', 'hangar interior',
    'harbor docks', 'marina pier', 'harbor water',

    # Grass / fields
    'grass field', 'green meadow', 'tall grass', 'lawn grass', 'grassy hillside',
    'pasture field', 'prairie grass', 'savanna grassland', 'grassy roadside',
    'overgrown grass', 'park lawn',

    # Trees / foliage
    'tree branches', 'leafy canopy', 'forest background', 'forest understory',
    'bush and shrubs', 'fallen leaves on ground', 'autumn leaves', 'fall foliage',
    'maple leaf carpet',

    # Rock / stone
    'rocky ground', 'gravel ground', 'stony path', 'rock pile', 'stone wall',
    'masonry wall', 'cliff face', 'boulder field', 'canyon rocks', 'quarry rocks',
    'rocky shoreline', 'pebble beach',

    # Water / coast
    'open water', 'ocean waves', 'sea surface', 'sea horizon', 'river water',
    'lake water', 'pond surface', 'rippled water', 'water reflections',
    'foamy waves', 'shoreline surf', 'coastal shoreline', 'beach sand',
    'kelp and seaweed',

    # Built / scene context
    'building facade', 'city street', 'mountain landscape', 'garden background',
    'playground background', 'stadium seats', 'campsite background', 'campsite ground',
    'picnic table', 'flower bed', 'garden bed', 'farm field', 'crop rows',
    'barn interior', 'pasture fence', 'straw hay', 'muddy ground', 'wooden floor',
    'carpet floor', 'countertop surface', 'wooden table', 'zoo enclosure fence',
    'cage mesh', 'rock outcrop', 'snow ground', 'desert sand',
]


# class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#                    'bus', 'car', 'cat', 'chair', 'cow',
#                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
#                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
#                    ]

class_names = [
    'airplane', 'butterfly', 'clock', 'dog', 'football', 'gun', 'kangaroo', 'monkey', 'pumpkin', 'seal', 'squirrel', 'train',
    'bear', 'cactus', 'corn', 'dolphin', 'fox', 'hat', 'lifeboat', 'motorcycle', 'rabbit', 'sheep', 'sunflower', 'truck',
    'bicycle', 'car', 'cow', 'elephant', 'frog', 'helicopter', 'lion', 'ostrich', 'racket', 'ship', 'tent', 'umbrella',
    'bird', 'cat', 'crab', 'fishing rod', 'giraffe', 'horse', 'lizard', 'owl', 'sailboat', 'shrimp', 'tiger', 'wheat',
    'bus', 'chair', 'crocodile', 'flower', 'goose', 'hot air balloon', 'mailbox', 'pineapple', 'scooter', 'spider', 'tortoise', 'wolf',
]

                   
# new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
#                    'bus', 'car', 'cat', 'chair seat', 'cow',
#                    'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
#                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
#                    ]
new_class_names = [
    'airplane', 'butterfly', 'clock', 'dog', 'football', 'gun', 'kangaroo', 'monkey', 'pumpkin', 'seal', 'squirrel', 'train',
    'bear', 'cactus', 'corn', 'dolphin', 'fox', 'hat', 'lifeboat', 'motorcycle', 'rabbit', 'sheep', 'sunflower', 'truck',
    'bicycle', 'car', 'cow', 'elephant', 'frog', 'helicopter', 'lion', 'ostrich', 'racket', 'ship', 'tent', 'umbrella',
    'bird', 'cat', 'crab', 'fishing rod', 'giraffe', 'horse', 'lizard', 'owl', 'sailboat', 'shrimp', 'tiger', 'wheat',
    'bus', 'chair', 'crocodile', 'flower', 'goose', 'hot air balloon', 'mailbox', 'pineapple', 'scooter', 'spider', 'tortoise', 'wolf',
]




class_names_coco = ['person','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee',
                    'skis','snowboard','sports ball','kite','baseball bat',
                    'baseball glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor','laptop','mouse',
                    'remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hair drier','toothbrush',
]

new_class_names_coco = ['person with clothes,people,human','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack,bag',
                    'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessertspoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
                    ]


BACKGROUND_CATEGORY_COCO = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge',
                        ]
