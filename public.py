from flask import *
# from database import *

public=Blueprint('public',__name__)

@public.route('/')
def home():
	return render_template("home.html")

# @public.route('/login',methods=['get','post'])
# def login():
# 	if 'submit'	in request.form:
# 		usname=request.form['username']
# 		pswd=request.form['password']
# 		# print(usname,pswd)
# 		q="SELECT * FROM login WHERE username='%s' AND password='%s'"%(usname,pswd)
# 		res=select(q)
# 		if res:
# 			session['lid']=res[0]['login_id']
# 			if res[0]['user_type']=='admin':
				
# 				return redirect(url_for('admin.adminhome'))

# 			elif res[0]['user_type']=='doctor':
# 				q="select * from doctors where login_id='%s'"%(session['lid'])
# 				res=select(q)
# 				session['doctor_id']=res[0]['doctor_id']

# 				return redirect(url_for('doctor.dr_home'))

# 			elif res[0]['user_type']=='user':
# 				q="select * from users where login_id='%s'"%(session['lid'])
# 				res=select(q)
# 				session['user_id']=res[0]['user_id']


# 				return redirect(url_for('user.us_home'))
# 	return render_template("login.html")

# @public.route('/dr_register',methods=['get','post'])
# def dr_register():
# 	if 'submit' in request.form:
# 		fname=request.form['first_name']
# 		lname=request.form['last_name']
# 		hname=request.form['housename']
# 		plc=request.form['place']
# 		lndmrk=request.form['landmark']
# 		eductn=request.form['qualification']
# 		phn=request.form['phone']
# 		mail=request.form['email']
		
# 		duname=request.form['username']
# 		pswd=request.form['password']

# 		a="insert into login values(null,'%s','%s','pending')"%(duname,pswd)
# 		id=insert(a)

# 		b="insert into doctors values(null,'%s','%s','%s','%s','%s','%s','%s','%s','%s','pending')"%(id,fname,lname,hname,plc,lndmrk,eductn,phn,mail,sts)
# 		insert(b)

# 		return redirect(url_for('public.login'))
# 	return render_template("dr_register.html")

# @public.route('/user_register',methods=['get','post'])
# def user_register():
# 	if 'submit' in request.form:
# 		fname=request.form['first_name']
# 		lname=request.form['last_name']
# 		hname=request.form['house_name']
# 		plc=request.form['place']
# 		phone=request.form['phn']
# 		mail=request.form['email']
# 		uuname=request.form['username']
# 		pswd=request.form['password']

# 		c="insert into login values(null,'%s','%s','user')"%(uuname,pswd)
# 		id=insert(c)

# 		d="insert into users values(null,'%s','%s','%s','%s','%s','%s','%s')"%(id,fname,lname,hname,plc,phone,mail)
# 		insert(d)
# 		return redirect(url_for('public.login'))
# 	return render_template("user_register.html")