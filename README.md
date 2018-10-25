
run service using docker

`docker run -p 5000:5000 linekin/delf_service`

call service using `httpie`

`http localhost:5000/match file@test_images/`

api url path is `/match`. 
make sure the images file size is less than 1MB,
otherwise out of memory error may occurred. 


# TODO
* use `tensorflow serving`
* use `twisted`
* providing docker entrypoint to extract feature 
* resize large image file