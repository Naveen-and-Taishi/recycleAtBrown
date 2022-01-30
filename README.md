# recycleAtBrown
Our project for Hack@Brown. Received Wolfram Award. 
## Inspiration

In the spirit of Greenhouse, the theme of this year's hackathon, we, two sophomores studying CS at Brown, wanted to create a program that can make recycling a better experience. Recycling has become an integral part of our society's waste management ecosystem, and we as community members are tasked with playing our part in ensuring our trash go to the correct category.

## What it does

We trained a neural net to categorize items as one of six categories: cardboard, glass, metal, paper, plastic, and trash.

## How we built it

We used a deep learning tool called Lobe to train a neural network to correctly categorize items into the six categories, and used tensorflowjs to hook it up to our frontend with React.

## Challenges we ran into

As you can see from the repo, our original plan was to use our own CNN architecture to train the classifier, but unfortunately, we struggled with our lack of computing resources and time to properly train our model. We tried different architectures, including our own CNN architecture, ResNet, EfficientNet, and so forth, but we couldn't debug everything in time. So, the obvious next step for us is to make our model to work with our goal of higher than 80% using our own model.

## Accomplishments that we're proud of

As first time hackathoners, we were able to create a viable product in such a short time, and we efficiently shifted our strategy in terms of using DL to use a pre-trained model for the sake of the project.

## What we learned

After struggling to get our original model to give significant results, we saw firsthand roadblocks of both compute and time as precious resources for training a deep learning model. But more importantly, I think we learned the beauty of creating a product that can help other people, especially locally.

## What's next for Recycle@Brown

As stated above, we want to use our own neural network model to categorize the items. We also think there's hardware elements of this project that could be revamped to make the user experience much easier.

## Built With

    python
    react
    tensorflow

****
