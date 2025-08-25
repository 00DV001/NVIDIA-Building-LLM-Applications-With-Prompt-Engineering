from langchain_nvidia_ai_endpoints import ChatNVIDIA

def sprint(stream):
    for chunk in stream:
        print(chunk.content, end='')

base_url = 'http://llama:8000/v1'
model = 'meta/llama-3.1-8b-instruct'
llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

prompt = 'Tell me about cakes.'
sprint(llm.stream(prompt))

''' Output = Cakes! Who can resist the allure of a delicious, moist, and sweet cake? Here's a comprehensive overview:

**History of Cakes**

 have been a staple in many cultures for thousands of years. The ancient Egyptians, Greeks, and Romans all baked cakes as offerings to their gods and as a form of celebration. The first cakes were likely made from crushed grains, honey, and nuts. The modern cake, however, is believed to have originated in Europe during the Middle Ages.

**Types of Cakes**

, each with its own unique characteristics and flavors. Here are some popular ones:

1. **Butter Cake**: A classic cake made with butter, sugar, eggs, and flour.
 airy cake made with eggs, sugar, and flour.
 Cake**: A dense and rich cake made with a pound each of four basic ingredients: flour, butter, sugar, and eggs.
**: A creamy and rich dessert made with a mixture of cream cheese, sugar, eggs, and vanilla.
 with layers of sponge cake, cream, and fruit.
au**: A French cake made with layers of cake, buttercream, and fruit.
 and moist texture.A ring-shaped cake with a dense
 topped with cinnamon and sugar.d-like cake often
 cake made with chocolate, marshmallows, and graham crackers.
 **Cake Pops**: Bite-sized cakes on a stick, often dipped in chocolate and decorated.

**Cake Decorations**

 of toppings and fillings, including:ety

 sweet and creamy topping made with butter, sugar, and milk.
Glaze**: A thin, sweet topping made with powdered sugar and milk.
3. **Sprinkles**: Small, colorful decorations made from sugar or candy.
 other sweet treats added on top of the cake.or
**: Custom designs or images made from sugar or wafer paper.

 Celebrations**

Cakes are often associated with special occasions, such as:

1. **Birthdays**: A traditional way to celebrate a person's birthday.
 their guests.*: A sweet treat for the happy couple and
 special milestone in a relationship.
 often baked and decorated for holidays like Christmas, Easter, and Halloween.

Cake Fun Facts**

,000 pounds and took 12 hours to bake.5
 cake toppers were made from sugar and were used in ancient Egypt.
. Cake is a popular dessert in many cultures, with over 2 billion cakes consumed worldwide each year.
, followed closely by chocolate.is vanilla

 your sweet tooth! Do you have a specific type of cake or question in mind? '''

target_address = """\
Some Company
12345 NW Green Meadow Drive
Portland, OR 97203"""

prompt = '''\
Rewrite the prompt as it is without any changes:

Some Company
12345 NW Green Meadow Drive
Portland, OR 97203'''

llm_address = llm.invoke(prompt).content
print(llm_address)

''' output = Some Company
12345 NW Green Meadow Drive
Portland, OR 97203 '''

llm_address == target_address
# output = True