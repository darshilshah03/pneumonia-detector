const express = require('express')
const exphbs = require('express-handlebars');
const bodyParser = require('body-parser');

const multer = require('multer')

const app = express()

const storage = multer.diskStorage({
    destination : function(req,file,cb){
        cb(null,'./images/')
    },
    filename : function(req,file,cb){
        cb(null,file.originalname);
    }
});
var upload = multer({storage:storage})

app.engine('handlebars', exphbs());
app.set('view engine', 'handlebars');

app.use(bodyParser.urlencoded({extended:false}));
app.use(bodyParser.json());


app.get('/https://darshilshah03.github.io/pneumonia-detector/', (req,res)=>{
    res.render('index')
})

app.post('/results',upload.single('xrayPic') ,(req,res) => {
    console.log(req.file)

    const spawn = require('child_process').spawn
    const pythonProcess = spawn('python3',['./predict.py' , req.file.originalname])

    pythonProcess.stdout.on('data', (data) => {
        console.log(data.toString())
        
        if(data>0.5)
            res.send('The person does not have Pneumonia.')
        else
            res.send('The person has Pneumonia.')
    })  
    

    pythonProcess.stderr.on('data',(data) => {
        console.error(data.toString())
    })
    
})

app.listen(3000,() => {
    console.log("Server started at port 3000")
})

