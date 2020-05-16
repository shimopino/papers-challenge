# Tips for Git

### rename directory

```sh
$ git mv <previous name> <new name>
```

- [in-a-git-repository-how-to-properly-rename-a-directory](https://stackoverflow.com/questions/11183788/in-a-git-repository-how-to-properly-rename-a-directory)

### remove directory from git repo

```sh
# delete directory from local filesystem
$ git rm -r <directory you remove>

# commit
$ git commit . -m "remove directory"

# push
$ git push origin <your branch>
```

```sh
# remove directory only from git repo
$ git rm -r --cached <directory in git repo>
```

- [How to remove a directory from git repository?]()https://stackoverflow.com/questions/6313126/how-to-remove-a-directory-from-git-repository
