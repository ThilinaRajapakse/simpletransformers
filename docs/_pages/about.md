---
permalink: /about/
title: "About"
excerpt: "Simple Transformers is the easy to use but incredibly powerful Transformers library."
toc: true
last_modified_at: 2020-05-02 16:20:14
---

Simple Transformers is a Natural Language Processing (NLP) library designed to simplify the usage of Transformer models without having to compromise on utility. It is built on the amazing work of Hugging Face and their [Transformers](https://github.com/huggingface/transformers) library.


[Install Simple Transformers]({{ "/docs/installation" | relative_url }}){: .btn .btn--success .btn--large}

## Why Simple Transformers?

Simple Transformers is designed around the way a person will typically use a Transformers model. At the highest level, Simple Transformers is branched into common NLP tasks such as text classification, question answering, and language modeling. Each of these tasks have their own task-specific Simple Transformers model. While all of the task-specific models maintain a consistent usage pattern *(initialize, train, evaluate, predict)*, this separation allows the freedom to adapt the models to their specific use case. To list but a few of the benefits;

- Input data formats are optimized for the task
- Outputs are clean and ready-to-use for the task with minimal to no post-processing required
- Unique configuration options for each task, while sharing a large, common base of configuration options across all tasks
- No boilerplate code attempting to squeeze together things that don't fit (Simple Transformers scripts rarely need to be longer than a few lines)
- Common sense defaults to get started quickly, so that you can configure as little *or* as much as you want to

Most of the time, you'll find that you only need to change the data files when switching between projects. This also means that Simple Transformers scripts are clear and readable, even to people unfamiliar with Transformers. If you are just starting out with Natural Language Processing, or if you are coming from a different field, this should help to get you up and running quickly without being overwhelmed.

In short, Simple Transformers gives you everything you need for your Transformer-based research, project, or product, and then gets out of your way.

Happy [transforming](/docs/installation/)!

---

## Credits

### Tools and Libraries

- [Transformers](https://github.com/huggingface/transformers) -- by Hugging Face. You guys rock!
- [Weights & Biases](https://www.wandb.com/) -- For tracking runs, and looking good while doing it!

### Docs Theme

- [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes) -- designed, developed, and maintained by Michael Rose.
- [The Noun Project](https://thenounproject.com) -- Garrett Knoll, Arthur Shlain, and [tracy tam](https://thenounproject.com/tracytam)
- [Font Awesome](http://fontawesome.io/)
- [Unsplash](https://unsplash.com/)
- [Jekyll](https://jekyllrb.com/)
- [jQuery](https://jquery.com/)
- [Susy](http://susy.oddbird.net/)
- [Breakpoint](http://breakpoint-sass.com/)
- [Magnific Popup](http://dimsemenov.com/plugins/magnific-popup/)
- [FitVids.JS](http://fitvidsjs.com/)
- Greedy Navigation - [lukejacksonn](https://codepen.io/lukejacksonn/pen/PwmwWV)
- [jQuery Smooth Scroll](https://github.com/kswedberg/jquery-smooth-scroll)
- [Lunr](http://lunrjs.com)

---

## Donations and Support

If you find Simple Transformers helpful, please consider becoming a Patron!

<a href="https://www.patreon.com/bePatron?u=20014970" data-patreon-widget-type="become-patron-button">Become a Patron!</a><script async src="https://c6.patreon.com/becomePatronButton.bundle.js"></script>

---

Simple Transformers is developed, maintained, and updated by Thilina Rajapakse with the help of all these wonderful [contributors](https://github.com/ThilinaRajapakse/simpletransformers#contributors-)!
