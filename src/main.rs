#![allow(nonstandard_style)]

use std::io::{self,BufRead};
use std::collections::HashMap;

static DISABLE_OUTPUT: bool = false;

#[derive(clap::Parser,Default,Debug)]
#[clap(author,version,about,long_about=None)]
struct Args
{
	#[arg(short='x',long,help="Ignore paths containing this substring")]
	exclude: Vec<String>,
	#[arg(short,long="time",help="Print phase timings to stderr")]
	timings: bool,

	#[arg(name="left",required=true,help="Left-hand (before) tree or log file")]
	treeL: String,
	#[arg(name="right",required=true,help="Right-hand (after) tree or log file")]
	treeR: String,
}

#[derive(Eq,Hash,PartialEq,Clone,Copy)]
struct Hash
{
	bytes: [u8;16],
}

impl Hash
{
	fn new(from: &str) -> Self
	{
		debug_assert_eq!(32,from.len());

		let result: std::mem::MaybeUninit::<[u8;16]> = std::mem::MaybeUninit::uninit();

		Hash
		{
			bytes: unsafe
			{
				let mut danger = result.assume_init();
				hex::decode_to_slice(from,&mut danger).unwrap();
				danger
			}
		}
	}
}

impl std::fmt::Debug for Hash
{
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
	{
		write!(f,"[{}]",hex::encode(self.bytes))
	}
}

struct ScopeTimer
{
	start: std::time::Instant,
	context: Option<&'static str>,
}

impl ScopeTimer
{
	fn new(enable: bool,context: &'static str) -> Self
	{
		ScopeTimer
		{
			context: if enable {Some(context)} else {None},
			start: std::time::Instant::now(),
		}
	}
}

impl Drop for ScopeTimer
{
	fn drop(&mut self)
	{
		if self.context.is_none() {return}

		let finish = self.start.elapsed();

		eprintln!(
			"{}: {:.3} ms",
			self.context.unwrap(),
			(finish.as_micros() as f32)/1000f32);
	}
}

#[derive(Clone)]
struct FSNode
{
	hash: Hash,
	path: &'static str,
	done: std::cell::Cell<bool>,
}

struct Side
{
	paths: Vec<&'static FSNode>,
}

struct Object
{
	by: u64,
	sides: [Side;2],
}

type MapPaths = HashMap<&'static str,&'static FSNode>;
type MapHashes = HashMap<Hash,Object>;

struct RKD
{
	sides: Vec<MapPaths>,
	hashes: MapHashes,
}

fn fsnode_open(path: &str) -> Box<dyn std::io::Read>
{
	use std::process::{Command,Stdio};

	if "-"==path
		{return Box::new(io::stdin().lock());}

	if !std::fs::metadata(path).unwrap().is_dir()
		{return Box::new(std::fs::File::open(path).unwrap());}

	let mut rk = Command::new("sudo");
	rk.args(["rk","-urei",path]);
	rk.stdin(Stdio::null());
	rk.stdout(Stdio::piped());

	Box::new(rk.spawn().unwrap().stdout.take().unwrap())
}

#[macro_use]
extern crate lazy_static;

lazy_static!
{
	static ref args: Args = <Args as clap::Parser>::parse();
}

fn slurp_log(stream: Box<dyn std::io::Read>) -> Vec<&'static str>
{
	let _timer = ScopeTimer::new(args.timings,"slurp_log");

	let mut log: Vec<&str> = vec!();

	for line in std::io::BufReader::new(stream).lines()
	{
		log.push(Box::leak(line.unwrap().into_boxed_str()));
	}

	log
}

fn main()
{
	if "-" == args.treeL  &&  "-" == args.treeR
	{
		eprintln!("Cannot compare stdin with itself!");
		std::process::exit(3);
	}

	// Pre-open both early so we can fail fast on bad arguments
	let (fileL,fileR) = (fsnode_open(&args.treeL),fsnode_open(&args.treeR));

	let (logL,logR) = (slurp_log(fileL),slurp_log(fileR));

	let mut rkd = RKD::new();

	// exit() is here to prevent destructors from being run, which adds a second or two to runtime
	std::process::exit(
		rkd.diff(&logL,&logR));
}

enum FSOp<'a>
{
	Delete,
	Create,
	CopyMove {src: &'a FSNode},
	Modify   {lhs: &'a FSNode},
}

impl FSNode
{
	fn new(path: &'static str,hash: Hash) -> Self
	{
		FSNode
		{
			path,
			hash,
			done: std::cell::Cell::new(false),
		}
	}

	fn new_or_recycle(path: &'static str,hash: Hash,otherSide: &mut MapPaths) -> Self
	{
		if let Some(o) = otherSide.get(path)
		{
			if hash == o.hash
			{
				// Identical to an item on the other side: recycle and set done
				o.set_done();
				return (*o).clone();
			}
		}

		FSNode::new(path,hash)
	}

	fn is_done(&self) -> bool
	{
		self.done.get()
	}

	fn set_done(&self)
	{
		debug_assert!(!self.is_done());
		self.done.set(true);
	}

	fn report(&self,disable: bool,op: &FSOp)
	{
		if self.is_done() {return};

		let mut lock = std::io::stdout().lock();

		use io::Write;

		match op
		{
			FSOp::Delete | FSOp::Create  =>
			{
				if !disable
				{
					let verb  =  if let FSOp::Delete = op {"RM"} else {"CR"};

					writeln!(
						lock,
						"{verb} {}",
						self.path).unwrap();
				}
			},
			FSOp::CopyMove{src} =>
			{
				assert_ne!(src.path,self.path);

				// If paths match, neither must be done and it's a MV
				let copy = if src.is_done() {true} else {src.set_done(); false};

				let verb  =  if copy {"CP"} else {"MV"};

				if !disable
				{
					let len=prefix_match_len(src.path.chars(),self.path.chars()); // Find common path prefix

					// Get rid of any common terminal-name prefix match to get a valid ancestor path
					let pos = match self.path[0..len].rfind('/')
					{
						Some(x) => 1+x,
						None => 0,
					};

					// Print common ancestor and then each path relative to that
					writeln!(
						lock,
						"{verb} '{}'\t'{}'\t'{}'",
						&self.path[0..pos],
						&src.path[pos..],
						&self.path[pos..]).unwrap();
				}
			},
			FSOp::Modify{lhs} =>
			{
				if !disable
				{
					writeln!(
						lock,
						"MD {}",
						self.path).unwrap();
				}

				lhs.set_done();
			},
		};

		self.set_done();
	}
}

impl Side
{
	fn new() -> Self
	{
		Self{paths: Vec::new()}
	}
}

impl Object
{
	fn new(by: u64) -> Self
	{
		Self{by,sides: [Side::new(),Side::new()]}
	}
}

impl RKD
{
	fn new() -> Self
	{
		RKD
		{
			hashes: MapHashes::new(),
			sides: Vec::new(),
		}
	}

	fn diff(&mut self,logL: &Vec<&str>,logR: &Vec<&str>) -> i32
	{
		assert_eq!(self.sides.len(),0);

		self.parse_side(&logL,&args.exclude);
		self.parse_side(&logR,&args.exclude);

		assert_eq!(self.sides.len(),2);

		self.diff_cpmv();
		self.diff_remaining();

		0
	}

	fn diff_remaining(&self)
	{
		let _timer = ScopeTimer::new(args.timings,"diff_remaining");

		debug_assert_eq!(self.sides.len(),2);

		for (path,nodeL) in &self.sides[0]
		{
			if nodeL.is_done() {continue}

			let nodeR_ = self.sides[1].get(path);

			if nodeR_.is_none()
			{
				nodeL.report(DISABLE_OUTPUT,&FSOp::Delete);
				continue;
			}

			debug_assert_ne!(nodeL.hash,nodeR_.unwrap().hash);

			nodeL.report(DISABLE_OUTPUT,&FSOp::Modify{lhs: nodeR_.unwrap()});
		}

		for nodeR in self.sides[1].values()
		{
			nodeR.report(DISABLE_OUTPUT,&FSOp::Create);
		}
	}

	fn match_right<'a>(itL: &'a mut VecIterator<&FSNode>,nodeR: &FSNode) -> &'a FSNode
	{
		while let Some(nodeL) = itL.curr()
		{
			if nodeL.path.bytes().rev().ge(nodeR.path.bytes().rev()) {break}
			itL.advance();
		}

		let (prev,curr) = (itL.prev(),itL.curr());

		if let (Some(pu),Some(cu)) = (prev,curr)
		{
			return match best_prefix_match(nodeR.path.bytes().rev(),pu.path.bytes().rev(),cu.path.bytes().rev())
			{
				BestPrefixMatch::First => pu,
				_ => cu,
			};
		}

		prev.unwrap_or_else(|| curr.unwrap())
	}

	fn diff_cpmv(&self)
	{
		let _timer = ScopeTimer::new(args.timings,"diff_cpmv");

		debug_assert_eq!(self.sides.len(),2);

		for obj in self.hashes.values()
		{
			// Build a reference list for RHS, excluding done items (i.e. make a "to report on" list)
			let mut pathsR = obj.sides[1].paths.iter()
				.filter_map(|nodeR| if !nodeR.is_done() {Some(*nodeR)} else {None})
				.collect::<Vec<_>>();

			if pathsR.is_empty() {continue}

			// Build a reference list for LHS, including done items (i.e. make a "possible cp/mv sources" list)
			let mut pathsL = obj.sides[0].paths.iter().map(|nodeL| *nodeL).collect::<Vec<_>>();

			if pathsL.is_empty() {continue}

			// Sort both lists by path suffix for good cp/mv matching
			sort_revpath(&mut pathsR);
			sort_revpath(&mut pathsL);

			let mut itL = VecIterator::new(&pathsL); // Iterator for following the RHS item on the left (like merge sort)

			for nodeR in pathsR
			{
				let nodeL = Self::match_right(&mut itL,nodeR);
				nodeR.report(DISABLE_OUTPUT,&FSOp::CopyMove{src: nodeL});
			}
		}
	}

	fn make_node(&mut self,side: usize,path: &'static str,hash: Hash) -> FSNode
	{
		debug_assert!(side<2);

		if side>0
			{return FSNode::new_or_recycle(path,hash,&mut self.sides[0])}
		else
			{FSNode::new(path,hash)}
	}

	fn make_hash_entry<'a>(hashes: &'a mut MapHashes,hash: &Hash,by: u64) -> &'a mut Object
	{
		if !hashes.contains_key(&hash)
		{
			hashes.insert(
				hash.clone(),
				Object::new(by));
		}

		let result = hashes.get_mut(hash).unwrap(); // TODO elide lookup when inserting
		assert_eq!(by,result.by);
		result
	}

	fn parse_side(&mut self,log: &Vec<&str>,excludes: &[String])
	{
		assert!(self.sides.len() < 2);

		let _timer = ScopeTimer::new(args.timings,"parse_log");

		let side = self.sides.len();

		debug_assert!(side<2);

		let mut files = MapPaths::new();

		'line_parser: for line in log 
		{
			let parsed = LogLine::parse(&line).unwrap().1;

			for substr in excludes
			{
				if parsed.path.contains(substr)
				{
					continue 'line_parser;
				}
			}

			let node = Box::leak(
				Box::new(
					self.make_node(
						side,
						&parsed.path,
						parsed.hash.clone()))); // TODO improve

			files.insert(parsed.path,node);

			Self::make_hash_entry(&mut self.hashes,&parsed.hash,parsed.by).sides[side].paths.push(node);
		}

		self.sides.push(files);
	}
}

struct LogLine
{
	by: u64,
	hash: Hash,
	path: &'static str,
}

fn hexhash(input: &str) -> nom::IResult<&str,Hash>
{
	const count: usize = 32;

	use nom::
	{
		Err::Failure,
		error::ParseError,
		error::ErrorKind::*,
	};

	if input.len() < count
	{
		return Err(Failure(ParseError::from_error_kind(input,HexDigit)));
	}

	let (_,strHash) = nom::combinator::all_consuming(
		nom::character::complete::hex_digit1)(&input[0..count])?;

	Ok(
		(
			&input[count..],
			Hash::new(strHash),
		)
	)
}

impl LogLine
{
	fn parse(input: &str) -> nom::IResult<&str,Self>
	{
		use nom::{
			sequence::*,
			character::complete::*,
			bytes::complete::tag,
			combinator::{opt,all_consuming},
		};

		let (rest,fields) = all_consuming(
			tuple(
				(
					preceded(space0,u64),
					preceded(tag("  "),hexhash),
					preceded(tag("  "),preceded(opt(tag("./")),not_line_ending)),
				)
			)
		)(input)?;

		Ok((
			rest,
			LogLine
			{
				by: fields.0,
				hash: fields.1,
				path: unsafe_dup_str(fields.2),
			},
		))
	}
}

fn unsafe_dup_slice<T>(s: &[T]) -> &'static [T]
{
	unsafe
	{
		std::slice::from_raw_parts(s.as_ptr(),s.len())
	}
}

fn unsafe_dup_str(s: &str) -> &'static str
{
	std::str::from_utf8(unsafe_dup_slice(s.as_bytes())).unwrap()
}

////////////////////////////////////////////////////////////////////////////////

fn sort_revpath(v: &mut [&FSNode])
{
	v.sort_by(|l,r|
		l.path.as_bytes().iter().rev().cmp(r.path.as_bytes().iter().rev()));
}

enum BestPrefixMatch
{
	First,
	Second,
	Neither, // Tie
	// TODO distinguish tie from "no match" and implement a suffix-match threshold for the latter
}

fn best_prefix_match<I: Iterator<Item=u8>>(r: I,mut l1: I,mut l2: I) -> BestPrefixMatch
{
	// TODO consider tie-breaking Neither based on the lengths of the three strings

	for rb in r
	{
		let (l1match,l2match) = (Some(rb) == l1.next(),Some(rb) == l2.next());

		if l1match
		{
			if l2match {continue}; // l1 and l2 both match r so far

			return BestPrefixMatch::First; // l1 matches one more byte of r than l2
		}
		else
		{
			if !l2match {break}; // Tie: l1 and l2 mismatch with r at the same byte

			return BestPrefixMatch::Second; // l2 matches one more byte of r than l1
		}
	}

	// Falling out of the loop means that both candidates matched the full length of the reference

	BestPrefixMatch::Neither
}

fn prefix_match_len<I: Iterator<Item=char>>(mut l: I,mut r: I) -> usize
{
	let mut count = 0usize;

	while let (Some(lb),Some(rb)) = (l.next(),r.next())
	{
		if lb!=rb
			{break}

		count = 1+count;
	}

	return count;
}

////////////////////////////////////////////////////////////////////////////////

pub struct VecIterator<'a, Item> where Item : 'a
{
	vector: &'a [Item],
	index: isize,
}

impl<'a, Item> VecIterator<'a, Item>
{
	fn new(vector: &'a [Item]) -> VecIterator<'a, Item>
	{
		VecIterator { index: 0, vector }
	}
}

impl<'a, Item> VecIterator<'a, Item>
{
	fn advance(&mut self)
	{
		self.index += 1;
	}
}

impl<'a, Item> VecIterator<'a, Item>
{
	fn prev(&mut self) -> Option<&'a Item>
	{
		self.vector.get((self.index - 1) as usize)
	}

	fn curr(&mut self) -> Option<&'a Item>
	{
		self.vector.get(self.index as usize)
	}
}