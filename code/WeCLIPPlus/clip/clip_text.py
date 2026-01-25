
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

# BACKGROUND_CATEGORY = [
#     # NICO domains
#     'autumn', 'dim', 'grass', 'outdoor', 'rock', 'water',

#     # Sky / lighting
#     'open sky', 'cloudy sky', 'blue sky with clouds', 'twilight sky',
#     'low light background', 'dim lighting', 'night scene', 'shadowy background',
#     'backlit silhouette', 'indoor low light', 'poorly lit room', 'underexposed scene',
#     'neon night street',

#     # Roads / urban ground
#     'road surface', 'asphalt pavement', 'highway road', 'street intersection',
#     'traffic lane markings', 'sidewalk pavement', 'parking lot', 'parking lot lines',
#     'rural road',

#     # Transport infrastructure
#     'railway tracks', 'railway ballast', 'train platform', 'tunnel interior',
#     'airport runway', 'airport tarmac', 'hangar interior',
#     'harbor docks', 'marina pier', 'harbor water',

#     # Grass / fields
#     'grass field', 'green meadow', 'tall grass', 'lawn grass', 'grassy hillside',
#     'pasture field', 'prairie grass', 'savanna grassland', 'grassy roadside',
#     'overgrown grass', 'park lawn',

#     # Trees / foliage
#     'tree branches', 'leafy canopy', 'forest background', 'forest understory',
#     'bush and shrubs', 'fallen leaves on ground', 'autumn leaves', 'fall foliage',
#     'maple leaf carpet',

#     # Rock / stone
#     'rocky ground', 'gravel ground', 'stony path', 'rock pile', 'stone wall',
#     'masonry wall', 'cliff face', 'boulder field', 'canyon rocks', 'quarry rocks',
#     'rocky shoreline', 'pebble beach',

#     # Water / coast
#     'open water', 'ocean waves', 'sea surface', 'sea horizon', 'river water',
#     'lake water', 'pond surface', 'rippled water', 'water reflections',
#     'foamy waves', 'shoreline surf', 'coastal shoreline', 'beach sand',
#     'kelp and seaweed',

#     # Built / scene context
#     'building facade', 'city street', 'mountain landscape', 'garden background',
#     'playground background', 'stadium seats', 'campsite background', 'campsite ground',
#     'picnic table', 'flower bed', 'garden bed', 'farm field', 'crop rows',
#     'barn interior', 'pasture fence', 'straw hay', 'muddy ground', 'wooden floor',
#     'carpet floor', 'countertop surface', 'wooden table', 'zoo enclosure fence',
#     'cage mesh', 'rock outcrop', 'snow ground', 'desert sand',
# ]


BACKGROUND_CATEGORY = [
    # water / shoreline / coastal structures
    'pond','marsh','wetland','swamp','bay','harbor','port','dock','pier','wharf','jetty','breakwater',
    'shore','shoreline','coast','coastline','sandbar','mud','mudflat','tidal flat','tidepool','estuary',
    'delta','inlet','cove','lagoon','channel','canal','creek','brook','waterfall','rapids',
    'boat','canoe','kayak','raft','sailboat','motorboat','yacht','ship','barge','ferry',
    'buoy','lifeguard tower','lighthouse','seawall','boat ramp','marina','boardwalk','promenade',
    'ripples','foam','spray','mist','reflection','wet sand','driftwood','seashell','pebbles','gravel',

    # landforms / terrain
    'meadow','prairie','savanna','pasture','farmland','orchard','vineyard','grove','thicket','underbrush',
    'trail','path','footpath','dirt path','hiking trail','canyon','ravine','gorge','plateau','ridge',
    'dune','sand','muddy ground','clay','silt','stones','cobblestones','outcrop','scree','talus',
    'lava','volcanic rock','basalt','granite','cave','sinkhole','glacier','iceberg',
    'puddle','mud puddle','snowbank','slush',

    # vegetation (non-bird-related)
    'reeds','cattails','tall grass','weeds','shrubs','hedge','moss','lichen','algae','seaweed',
    'lilypad','water plants','flowers','wildflowers','fern','palm','palm tree','pine','evergreen',
    'log','stump','bark','roots','fallen tree','brush','mulch',

    # human-made outdoor stuff
    'dock posts','rope','net','fishing gear','cooler','bucket','crate','barrel','drum','trash can',
    'plastic bottle','litter','debris','drain','storm drain','culvert','ditch','irrigation',
    'bridge railing','guardrail','concrete','asphalt','pavement','sidewalk','curb','parking lot',
    'crosswalk','traffic light','streetlight','utility pole','power lines','telephone lines',
    'billboard','graffiti','construction','scaffolding','floodlight','fountain',
    'bench','picnic table','gazebo','playground','slide','swing set',
    'tent','campground','campfire','fire pit','grill','smoke',

    # buildings / urban contexts
    'barn','shed','cabin','hut','warehouse','factory','industrial site','dockyard',
    'apartment','condo','skyscraper','downtown','suburb','neighborhood','courtyard',
    'balcony','porch','stairs','doorway','window','brick','stone wall','wooden wall','metal fence',
    'chain-link fence','gate','railing','roof','chimney',

    # weather / sky / lighting
    'sunset','sunrise','twilight','dusk','dawn','shade','shadow','sun glare',
    'fog','haze','smog','rain','drizzle','storm','thunderstorm','lightning',
    'wind','overcast','blue sky','night','moonlight','streetlights',

    # vehicles and common objects
    'car','truck','pickup','van','bus','train','freight car','tractor','bulldozer','excavator',
    'bicycle','motorcycle','scooter','boat trailer','parking meter',
    'cone','barrier','signpost','mailbox','bench','table','chair','umbrella','towel','blanket',
    'backpack','bag','cooler','basket','bottle','cup',

    # other animals (non-bird)
    'dog','cat','horse','cow','goat','sheep','pig','deer','elk','moose',
    'bear','fox','coyote','wolf','raccoon','skunk','rabbit','hare','squirrel',
    'chipmunk','mouse','rat','beaver','otter','seal','sea lion','dolphin','whale',
    'turtle','lizard','snake','frog','toad','fish','crab','lobster','shrimp','clam',
    'snail','slug','butterfly','moth','bee','wasp','ant','beetle','dragonfly','mosquito',

    # indoor-ish / weird but plausible (zoos, parks, human spaces)
    'zoo enclosure','aquarium tank','glass','tile','concrete floor','wood floor','carpet',
    'cage','wire mesh','enclosure fence','handrail','poster','banner','painting',
]


import os

# class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#                    'bus', 'car', 'cat', 'chair', 'cow',
#                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
#                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
#                    ]

# _all_class_names = [
#     'airplane', 'butterfly', 'clock', 'dog', 'football', 'gun', 'kangaroo', 'monkey', 'pumpkin', 'seal', 'squirrel', 'train',
#     'bear', 'cactus', 'corn', 'dolphin', 'fox', 'hat', 'lifeboat', 'motorcycle', 'rabbit', 'sheep', 'sunflower', 'truck',
#     'bicycle', 'car', 'cow', 'elephant', 'frog', 'helicopter', 'lion', 'ostrich', 'racket', 'ship', 'tent', 'umbrella',
#     'bird', 'cat', 'crab', 'fishing rod', 'giraffe', 'horse', 'lizard', 'owl', 'sailboat', 'shrimp', 'tiger', 'wheat',
#     'bus', 'chair', 'crocodile', 'flower', 'goose', 'hot air balloon', 'mailbox', 'pineapple', 'scooter', 'spider', 'tortoise', 'wolf',
# ]

# _all_new_class_names = [
#     'airplane', 'butterfly', 'clock', 'dog', 'football', 'gun', 'kangaroo', 'monkey', 'pumpkin', 'seal', 'squirrel', 'train',
#     'bear', 'cactus', 'corn', 'dolphin', 'fox', 'hat', 'lifeboat', 'motorcycle', 'rabbit', 'sheep', 'sunflower', 'truck',
#     'bicycle', 'car', 'cow', 'elephant', 'frog', 'helicopter', 'lion', 'ostrich', 'racket', 'ship', 'tent', 'umbrella',
#     'bird', 'cat', 'crab', 'fishing rod', 'giraffe', 'horse', 'lizard', 'owl', 'sailboat', 'shrimp', 'tiger', 'wheat',
#     'bus', 'chair', 'crocodile', 'flower', 'goose', 'hot air balloon', 'mailbox', 'pineapple', 'scooter', 'spider', 'tortoise', 'wolf',
# ]
_all_class_names = ['bird']

_all_new_class_names = ['bird']

# Check for CLIP_TEXT_VERSION environment variable
_version = os.environ.get('CLIP_TEXT_VERSION', None)
if _version:
    # Convert underscores to spaces for class name lookup
    # (directory names use underscores, but class definitions use spaces)
    _version = _version.replace('_', ' ')

if _version and _version in _all_class_names:
    # Filter to single class if version matches a class name
    class_names = [_version]
    new_class_names = [_version]
else:
    # Use full list if no version specified or version not found
    class_names = _all_class_names
    new_class_names = _all_new_class_names




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
