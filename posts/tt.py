img_path = file.parent.parent / 'image'
    for file in img_path.glob('*'):
        print(file)
        
        
        
file.rename(file.with_name('index.md')
            

 if not pathlib.Path(newpath:=str(file).split('.')[0]).exists() :
        pathlib.Path(newpath).mkdir(parents=True, exist_ok=True)
    shutil.move(str(file), str(file).split('.')[0])