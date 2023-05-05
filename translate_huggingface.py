#!/usr/bin/env python
# coding: utf-8
import time

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import utils


def translate(text_list, src_lang, tgt_lang):
    global model
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", device_map='auto')
    global tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", device_map='auto')
    output_list = []
    for t in text_list:
        ans = translate_with_m2m100(t, src_lang, tgt_lang)
        output_list.append(ans)
    return output_list


# 保存模型,离线可用
# tokenizer.save_pretrained("./models/facebook/m2m100_418M")
# model.save_pretrained("./models/facebook/m2m100_418M")

def translate_with_m2m100(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_en = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_opt(text_list, src_lang, tgt_lang, chunk_size=100):
    chunks = utils.split_text(text_list, chunk_size)
    print(len(chunks), chunks)
    combine_list = []
    for c in chunks:
        l = "\n".join(c)
        combine_list.append(l)
    ans = translate(combine_list, src_lang, tgt_lang)
    return ans


if __name__ == '__main__':
    # t = "Hello, my dog is cute"

    text_list = ['Like almost there, almost.', "Okay, so let's revisit 3D from AI art.",
                 'And so I, like, control that, has gotten us a lot closer.',
                 'Okay, so to start off with, there is three keys.', 'So we have Nerf.',
                 'So Nerf is basically a way of doing 3D', 'can reconstruct with 2D images.',
                 "And the thing that's really nice about it", 'is that you can give it trash images',
                 "and like, it'll give you something back.", 'Like, it might not be a good thing back,',
                 'but it still does have some sort of reconstruction.', 'Whereas, like, other methods might just fail.',
                 "Nerf's always kind of have some type of output.", "And that's really nice,",
                 'because you can kind of do some back and forth stuff.',
                 'And like, so like, you put trash in, you get trash out,',
                 'and then you might fix that trash a little bit.', "Okay, so that's Nerf's.",
                 'Secondly, we have ControlNet.', 'ControlNet is a way of adding extra conditionals',
                 'into stable diffusion.', "And that's really important for our process.",
                 "It's like really important overall.", "But for what we're doing, we're doing, like,",
                 'constrained optimization, basically.', 'Like, we want the images to be optimized towards',
                 'a particular 3D object.', "And so it's really helpful for that.",
                 "So we're mostly going to be using the pose aspect of ControlNet.",
                 'ControlNet can do a bunch of different things.', 'So if we use depth, then it can use normal maps,',
                 'and scribbles, and any sort of thing to create a constraint,',
                 'and like hand poses, and all these other things.',
                 "So for us, we're just going to be using human poses.",
                 "So that's going to really help with making sure", "that arms and stuff don't disappear,",
                 'and that like, we have a consistent model throughout.',
                 'And yeah, it just, like, it works so much better', 'than I was expecting.',
                 "And then there's also textual embeddings and lures.", "And so that's to get consistent models.",
                 'Textual embeddings and lures are both ways', 'of constraining the actual visual information a lot.',
                 'And so we just grabbed some from the internet,', 'and yeah, that off to the races.',
                 'So one of the reasons why I wanted to revisit', 'this 3D Nerf stuff is because Google has recently',
                 'put out this Dreamboot 3D paper.', "And so basically how it's set up is they",
                 'have a three-step process.', 'They start with something that is like Dreamboot,',
                 'so which is very similar to lures or textual embeddings.', "So it's a personalized AI.",
                 "And then they're combining that with a Nerf.",
                 'But one of the things they found is that they were getting', 'really overbaked 3D objects.',
                 'So what I mean by overbaked in this sense', 'is that it has the essence of the object,',
                 "but it's kind of overdone.", "It's like you can tell it's an owl, but it's an owl.",
                 'So they were trying to find out ways around that.',
                 'And they decided to settle on an image-to-image translation,',
                 'which is really similar to what I had been previously trying.',
                 'So previously I had to use multi-view image-to-image', 'translation to kind of improve the image.',
                 "And so that's kind of what I have been previously trying.",
                 'So I really wanted to revisit this project',
                 'because the second step was basically what I had already done.',
                 'Their third step was just kind of returning all these things', 'back to the 3D Nerf stuff.',
                 "So that's kind of their pipeline.", 'Their pipeline is partial dream-booth 3D,',
                 'then an image-to-image translation,', 'and then revisiting the dream-booth 3D.',
                 "Yeah, so it's a three-step process.", 'So I kind of looked at this and I was like, okay,',
                 'so the problem that I was facing the first time around', 'is consistency.',
                 'All of my images were all over the place.',
                 'So I was like, maybe the thing I need to fix is making it,',
                 'that the images are a lot more consistent.',
                 'And so textual embeddings and lures are the obvious way', 'to do that.',
                 "But there's also control met.", 'And control met is a really powerful way of limiting',
                 'the kind of different types of images that stable diffusion', 'can create.',
                 'So by combining those two, my step is basically text-to-image', 'or image-to-image with control met',
                 'to kind of limit it to what a 3D object is already.',
                 'These steps we can also repeat over and over again.', 'So, and I was trying this out,',
                 'and sometimes it would converge and we get better,', 'and then sometimes it would get worse.',
                 "So it's something that I still need to figure out", 'at that part of it a little bit,',
                 'but it sometimes does get better,', 'which is kind of interesting to think about.',
                 'If you wanted to actually recreate the paper more faithfully,',
                 'there is an open source version of Dream Booth',
                 "and there's an open source version of Dream Fusion.", 'And so you actually can recreate this all.',
                 'It would actually take a fair amount of work.', 'And the open source version of Dream Fusion',
                 'requires a larger GPU than I currently have.', 'So I also wanted to point out,',
                 'the Google one does have much better metrics.',
                 'If you look at their things like depth maps and stuff,',
                 'their depth maps just look a lot cleaner than mine,', 'mine, or a little bit like,',
                 "there's some patches, them, they're not, like,", "they're not analytically good, I guess.",
                 "Okay, so let's look a little bit at the results.", 'So in particular, when we look at the GUI nerfs,',
                 'we see that the images are actually a little bit overfitted.', "So if you're near one of the points",
                 'that one of the pitchers or viewpoints,', "you'll have pretty good coloration,",
                 'but as you move away from that,', "it'll keep a lot of the structure,",
                 'but a lot of the actual coloration will fall off.',
                 'And so that had a little bit to do with how I did the sampling.', 'I was a little bit lazy,',
                 'and I used to do a circular camera.',
                 'I should be doing some more random sampling and things like that.',
                 "So that's something that was a little bit lazy on my view.",
                 "The other reason for doing that is I've noticed a lot of people",
                 'using destuttering to actually clean up video.', 'So if you can destutter between the images,',
                 'maybe that would give a more consistent image.', 'So that was the other thinking',
                 'between making a circular video is like,', 'maybe in that case, we can use destuttering',
                 'to like further reduce the noise.', 'I never got around to that.',
                 'Secondly, we can look at the models that they output.', 'So nerfs are a complete 3D object,',
                 "but oftentimes if you're using them in some sort of game,", 'you want a standard mesh.',
                 'You can use things like marching cube algorithms', 'to get a mesh from the 3D.',
                 'These meshes are pretty patchy,', 'but like you can see her ponytail,',
                 'like it captured a ponytail.', "So like there's a lot of detail that they have as well.",
                 "And hopefully there's like some method", 'of more encompassing some of this.',
                 'Like I think it has a lot to do with thresholds',
                 'on which part of the cloud density actually gets captured.',
                 'So I think there are ways of kind of filling out',
                 "some of these holes, although there's still a lot", 'of work to be done there.',
                 'So but like a ponytail.', 'One of the craziest sentences from the Dream Blue 3D paper is,',
                 'a key insight at this stage is that Dream Blue', 'can effectively generate unseen views of a subject',
                 'given that the initial images are close enough', 'to the unseen views.',
                 'So just like just take a moment, think about that.',
                 'It sort of means that like some of these algorithms', 'might have a version of 3D,',
                 'like they might have through like a sense of 3D built into them.', 'And that is so interesting.',
                 "Like that's not something that they were designed", 'to have a 3D understanding of the world,',
                 'but like it kind of brings up this topic', 'that comes up sometimes in AI with a Misa optimizers.',
                 "And that's basically this idea", "that there's these like an optimization algorithm",
                 'might create these sub-optimization algorithms', 'to help them solve problems,',
                 "even though they weren't tasked with these sort of designations.",
                 'Internally they might end up building out some of these systems.',
                 "If you're trying to do this denoising of 2D images,",
                 'you might actually need to have an understanding', 'of the 3D world so that to help you out',
                 'like with that denoising aspect.', 'And this is like, this is subtle proof',
                 'that that might be a thing.', "Like it's not, this is evidence", 'that that might be the case.',
                 "It's not a given fact like you would need to do", 'a lot more statistics and stuff,',
                 'but it just kind of like, to me,', "it shows that there's like a lot more going on here",
                 'than sometimes people assume.', 'Like a lot of people have just been running away',
                 'with this idea that like,', "oh, they're just like a collaging cool",
                 "and it's just mixing images together.", 'And this is like, this is giving it the idea',
                 'that maybe it actually has a 3D understanding of the world',
                 "that it's gained through trying to denoise.", "And that's like, that's really fascinating.",
                 "That kind of idea that there's like sub-optimizations", 'that happen within some of these algorithms',
                 'to help them further understand.', "So it's like, it kind of is like another piece of evidence",
                 'in that learning bucket where they are not doing', 'the kind of really simplistic thing',
                 "that people seem to be stating that they're doing,", "but they're doing something more interesting.",
                 "And to me, that's just fascinating.", 'I want to do include a bunch of the tools',
                 'that I was using.', 'So a lot of them are just scripts', "that I've put into this box here.",
                 "It was a lot of scripting and that's yeah.", "Yeah, that's it.", 'Have a great day.',
                 'And like if you try anything out,', 'I really want to see you.']

    print(len(text_list))
    # start = time.time()
    # ans = translate_opt(text_list, "en", "zh")
    # print(f'Time for translate: {time.time() - start}s')
    # print()
    # print(ans)
# t = translate_with_m2m100(t, "en", "zh")
# print(t)
