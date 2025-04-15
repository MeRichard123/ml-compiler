import os, random

phrases = ['Light the spark','Time flies',"Never settle","Stay curious","Keep moving","Lost in thought","Seize the day","Brave the storm","Beyond the stars","Take the leap","Dream big","Follow through","Speak truth","Listen closely","Keep it simple","Silence speaks","Rise again","All in","Just breathe","Know your worth","Stay wild","Find your fire","Let it be","Run free","Change is good","No regrets","Always learning","Push forward","Face the fear","Do it anyway","Calm within","Trust the path","Stay gold","Be the light","Fuel the fire","Take your time","In the moment","Make it count","Onward now","Stay humble","Stay grounded","Think big","No turning back","Almost there","Leave a mark","Believe more","Stay awake","Open mind","Embrace change","Less is more","Go deeper","Keep exploring","Look ahead","No excuses","Be bold","Chase light","Be still","Watch closely","Live fully","Feel alive","Keep going","Don’t look back","Soft heart","Sharp mind","Move gently","Find your pace","Love louder","Walk tall","Speak softly","Begin again","Peace within","Walk your way","No fear","Be present","Stay strong","Lift others","Stay awake","Own it","Let go","Think slow","Speak kind","Show up","Go wild","Do good","Stay light","Dig deep","Stay real","All ears","Roll with it","Stay sharp","Hold fast","Stay close","Keep rising","Know peace","Feel more","Keep faith","Trust yourself","Stay ready","Take heart","Live true"]

random_words = [
    "apple", "journey", "crisp", "fly", "texture", "radiant", "spin", "border", "gleam", "whisper",
    "river", "climb", "soft", "battle", "frost", "ignite", "draw", "clever", "giant", "thunder",
    "drift", "lens", "marvel", "whisper", "bold", "glitch", "twist", "humble", "spark", "velvet",
    "dive", "puzzle", "breeze", "crown", "sprint", "launch", "vivid", "swirl", "glow", "stumble",
    "lantern", "roar", "maze", "silk", "forge", "moss", "quake", "breeze", "flicker", "anchor",
    "chant", "trace", "leap", "nimble", "vanish", "dusk", "jolt", "coil", "blade", "frost",
    "shimmer", "vault", "grasp", "flare", "tremble", "quest", "flare", "soothe", "crystal", "climb",
    "orbit", "dazzle", "bloom", "soak", "echo", "tinker", "hush", "bolt", "blink", "scatter",
    "vine", "clutch", "bounce", "flick", "brave", "grasp", "hollow", "veil", "rush", "murmur",
    "howl", "flicker", "tide", "bound", "hush", "glint", "lurch", "glow", "flare", "sprint", "soar"
]

for word in random_words:
    path = os.path.join('Lua', 'Curriculum1', "if_"+ word + '.lua')
    with open(path, 'w') as f:
        if random.choice([True, False]):
            f.write("if true then\n")
        else:
            f.write("if false then\n")
        f.write("   print(\"" + word + "\")\n")
        f.write("else\n")
        f.write("   print(\"" + random.choice(phrases) + "\")\n")
        f.write("end\n")
    print(f"Created file: {path}")