All Python files here were originally converted from MatLab (.m) files with Libermate (libermate/).

Libermate can process most of the files as-is, but there are occaisional syntax that confuse it.  These are easily corrected - just go to the line number it reports an error and you may notice a small change in format will work.  For example, this reports an error:

```
    rectangle('Position', [0.01, 0.8, 0.28 ,0.1],'FaceColor','w','EdgeColor','none');
```

But just change the space before a comma in the 3rd number of the vector to this and it works:

```
    rectangle('Position', [0.01, 0.8, 0.28, 0.1],'FaceColor','w','EdgeColor','none');
```

