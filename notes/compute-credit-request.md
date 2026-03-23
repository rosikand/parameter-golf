750 chars... 

---


I have several ideas that are theoretically sound and would like to verify empirically. If given the compute, I'd do: (1) run and verify baseline, (2) run and verify current SOTA, (3) implement my ideas on top of (1) and/or (2). My main current idea is called "test-time distillation": 
- store a compressed model, algorithmically expand it at eval time ("bloom" phase), use the expanded version as a teacher, via distillation, to improve either (1) raw evaluation scoring, (2) TTT ability. 
- this motivated by the fact that eval compute is much larger than what naive inference requires (along time and space... i.e., 8xH100 is more memory than the 16mb artifact we are limited to). 


I will be documenting everything at: https://github.com/rosikand/parameter-golf/ and submitting a PR once a performant result is obtained. 


---

Note: I filled out the form a few minutes ago, but I believe I will need more compute for the amount of things I want to run, so I am requesting the "development grant" instead. Please ignore that request if I receive this grant instead. 
---
I have several ideas that are theoretically sound and would like to verify empirically. If given the compute, I'd do: (1) run and verify baseline, (2) run and verify current SOTA, (3) implement my ideas on top of (1) and/or (2). My main current idea is called "test-time distillation": 
- store a compressed model, algorithmically expand it at eval time ("bloom" phase), use the expanded version as a teacher, via distillation, to improve either (1) raw evaluation scoring, (2) TTT ability. 
- this motivated by the fact that eval compute is much larger than what naive inference requires (along time and space... i.e., 8xH100 is more memory than the 16mb artifact we are limited to). 

Some other small ideas I had: 



I will be documenting everything at: https://github.com/rosikand/parameter-golf/ and submitting a PR once a performant result is obtained. 

