
/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2013 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
#if !defined (SIMD_FUNCTIONS_H__)
#define SIMD_FUNCTIONS_H__

//
//
//
//inline int add_sat(int a, int b)
//{
//	return a+b;
//}
//inline int sub_sat(int a, int b)
//{
//	return a-b;
//}
//inline int min(int a, int b)
//{
//	return a < b ? a : b;
//}
//inline int max(int a, int b)
//{
//	return a > b ? a : b;
//}



/*
This header file contains inline functions that implement intra-word SIMD
operations, that are hardware accelerated on sm_3x (Kepler) GPUs. Efficient
emulation code paths are provided for earlier architectures (sm_1x, sm_2x)
to make the code portable across all GPUs supported by CUDA. The following 
functions are currently implemented:

vabs2(a)        per-halfword absolute value, with wrap-around: |a|
vabsdiffs2(a,b) per-halfword absolute difference of signed integer: |a - b|
vabsdiffu2(a,b) per-halfword absolute difference of unsigned integer: |a - b|
vabsss2(a)      per-halfword abs. value, with signed saturation: sat.s16(|a|)
vadd2(a,b)      per-halfword (un)signed addition, with wrap-around: a + b
vaddss2(a,b)    per-halfword addition with signed saturation: sat.s16 (a + b)
vaddus2(a,b)    per-halfword addition with unsigned saturation: sat.u16 (a+b)
vavgs2(a,b)     per-halfword signed rounded average: (a+b+((a+b)>=0)) >> 1
vavgu2(a,b)     per-halfword unsigned rounded average: (a + b + 1) / 2
vcmpeq2(a,b)    per-halfword (un)signed comparison: a == b ? 0xffff : 0
vcmpges2(a,b)   per-halfword signed comparison: a >= b ? 0xffff : 0
vcmpgeu2(a,b)   per-halfword unsigned comparison: a >= b ? 0xffff : 0
vcmpgts2(a,b)   per-halfword signed comparison: a > b ? 0xffff : 0
vcmpgtu2(a,b)   per-halfword unsigned comparison: a > b ? 0xffff : 0
vcmples2(a,b)   per-halfword signed comparison: a <= b ? 0xffff : 0
vcmpleu2(a,b)   per-halfword unsigned comparison: a <= b ? 0xffff : 0
vcmplts2(a,b)   per-halfword signed comparison: a < b ? 0xffff : 0
vcmpltu2(a,b)   per-halfword unsigned comparison: a < b ? 0xffff : 0
vcmpne2(a,b)    per-halfword (un)signed comparison: a != b ? 0xffff : 0
vhaddu2(a,b)    per-halfword unsigned average: (a + b) / 2
vmaxs2(a,b)     per-halfword signed maximum: max(a, b)
vmaxu2(a,b)     per-halfword unsigned maximum: max(a, b)
vmins2(a,b)     per-halfword signed minimum: min(a, b)
vminu2(a,b)     per-halfword unsigned minimum: min(a, b)
vneg2(a,b)      per-halfword negation, with wrap-around: -a
vnegss2(a,b)    per-halfword negation, with signed saturation: sat.s16(-a)
vsads2(a,b)     per-halfword sum of abs diff of signed: sum{0,1}(|a-b|)
vsadu2(a,b)     per-halfword sum of abs diff of unsigned: sum{0,1}(|a-b|)
vseteq2(a,b)    per-halfword (un)signed comparison: a == b ? 1 : 0
vsetges2(a,b)   per-halfword signed comparison: a >= b ? 1 : 0
vsetgeu2(a,b)   per-halfword unsigned comparison: a >= b ? 1 : 0
vsetgts2(a,b)   per-halfword signed comparison: a > b ? 1 : 0
vsetgtu2(a,b)   per-halfword unsigned comparison: a > b ? 1 : 0
vsetles2(a,b)   per-halfword signed comparison: a <= b ? 1 : 0 
vsetleu2(a,b)   per-halfword unsigned comparison: a <= b ? 1 : 0 
vsetlts2(a,b)   per-halfword signed comparison: a < b ? 1 : 0
vsetltu2(a,b)   per-halfword unsigned comparison: a < b ? 1 : 0
vsetne2(a,b)    per-halfword (un)signed comparison: a != b ? 1 : 0
vsub2(a,b)      per-halfword (un)signed subtraction, with wrap-around: a - b
vsubss2(a,b)    per-halfword subtraction with signed saturation: sat.s16(a-b)
vsubus2(a,b)    per-halfword subtraction w/ unsigned saturation: sat.u16(a-b)
  
vabs4(a)        per-byte absolute value, with wrap-around: |a|
vabsdiffs4(a,b) per-byte absolute difference of signed integer: |a - b|
vabsdiffu4(a,b) per-byte absolute difference of unsigned integer: |a - b|
vabsss4(a)      per-byte absolute value, with signed saturation: sat.s8(|a|)
vadd4(a,b)      per-byte (un)signed addition, with wrap-around: a + b
vaddss4(a,b)    per-byte addition with signed saturation: sat.s8 (a + b)
vaddus4(a,b)    per-byte addition with unsigned saturation: sat.u8 (a + b)
vavgs4(a,b)     per-byte signed rounded average: (a + b + ((a+b) >= 0)) >> 1
vavgu4(a,b)     per-byte unsigned rounded average: (a + b + 1) / 2
vcmpeq4(a,b)    per-byte (un)signed comparison: a == b ? 0xff : 0
vcmpges4(a,b)   per-byte signed comparison: a >= b ? 0xff : 0
vcmpgeu4(a,b)   per-byte unsigned comparison: a >= b ? 0xff : 0
vcmpgts4(a,b)   per-byte signed comparison: a > b ? 0xff : 0
vcmpgtu4(a,b)   per-byte unsigned comparison: a > b ? 0xff : 0
vcmples4(a,b)   per-byte signed comparison: a <= b ? 0xff : 0
vcmpleu4(a,b)   per-byte unsigned comparison: a <= b ? 0xff : 0
vcmplts4(a,b)   per-byte signed comparison: a < b ? 0xff : 0
vcmpltu4(a,b)   per-byte unsigned comparison: a < b ? 0xff : 0
vcmpne4(a,b)    per-byte (un)signed comparison: a != b ? 0xff: 0
vhaddu4(a,b)    per-byte unsigned average: (a + b) / 2
vmaxs4(a,b)     per-byte signed maximum: max(a, b)
vmaxu4(a,b)     per-byte unsigned maximum: max(a, b)
vmins4(a,b)     per-byte signed minimum: min(a, b)
vminu4(a,b)     per-byte unsigned minimum: min(a, b)
vneg4(a,b)      per-byte negation, with wrap-around: -a
vnegss4(a,b)    per-byte negation, with signed saturation: sat.s8(-a)
vsads4(a,b)     per-byte sum of abs difference of signed: sum{0,3}(|a-b|)
vsadu4(a,b)     per-byte sum of abs difference of unsigned: sum{0,3}(|a-b|)
vseteq4(a,b)    per-byte (un)signed comparison: a == b ? 1 : 0
vsetges4(a,b)   per-byte signed comparison: a >= b ? 1 : 0
vsetgeu4(a,b)   per-byte unsigned comparison: a >= b ? 1 : 0
vsetgts4(a,b)   per-byte signed comparison: a > b ? 1 : 0
vsetgtu4(a,b)   per-byte unsigned comparison: a > b ? 1 : 0
vsetles4(a,b)   per-byte signed comparison: a <= b ? 1 : 0
vsetleu4(a,b)   per-byte unsigned comparison: a <= b ? 1 : 0
vsetlts4(a,b)   per-byte signed comparison: a < b ? 1 : 0
vsetltu4(a,b)   per-byte unsigned comparison: a < b ? 1 : 0
vsetne4(a,b)    per-byte (un)signed comparison: a != b ? 1: 0
vsub4(a,b)      per-byte (un)signed subtraction, with wrap-around: a - b
vsubss4(a,b)    per-byte subtraction with signed saturation: sat.s8 (a - b)
vsubus4(a,b)    per-byte subtraction with unsigned saturation: sat.u8 (a - b)
*/

#define not_b32(res, a) res = ~a
#define mov_b32(res, a) res = a
#define and_b32(res, a, b) res = a & b
#define or_b32(res, a, b) res = a | b
#define xor_b32(res, a, b) res = a ^ b
#define shr_u32(res, a, b) res = a >> b
#define shl_u32(res, a, b) res = a << b
#define sub_u32(res, a, b) res = a - b
#define add_u32(res, a, b) res = a + b
#define add_sat_s32(res, a, b) res = add_sat(a, b)
#define sub_sat_s32(res, a, b) res = sub_sat(a, b)
#define set_ge_s32_s32(res, a, b) res = (a >= b ? 0xffffffff : 0)
#define set_le_s32_s32(res, a, b) res = (a <= b ? 0xffffffff : 0)
#define set_gt_s32_s32(res, a, b) res = (a > b ? 0xffffffff : 0)
#define set_lt_s32_s32(res, a, b) res = (a < b ? 0xffffffff : 0)
#define cvt_s32_s16_s(res, a) res =                ((int)((signed short)((unsigned short)(a & 0x0000ffff))))
#define cvt_s32_s16_u(res, a) res = ((unsigned int)((int)((signed short)((unsigned short)(a & 0x0000ffff)))))



//inline unsigned int vabs2(unsigned int a)
inline unsigned int vabs2(unsigned int a)
{
	unsigned int r;
    
	unsigned int m,s;      
	and_b32(  m,a,0x80008000); // extract msb
	and_b32(  r,a,0x7fff7fff); // clear msb
	shr_u32(  s,m,15);         // build lsb mask
	sub_u32(  m,m,s);          //  from msb
	xor_b32(  r,r,m);          // conditionally invert lsbs
	add_u32(  r,r,s);          // conditionally add 1
         
	return r;           // halfword-wise absolute value, with wrap-around
}

inline unsigned int vabsss2(unsigned int a)
{
	unsigned int r;
    
	unsigned int m,s;      
	and_b32(  m,a,0x80008000); // extract msb
	and_b32(  r,a,0x7fff7fff); // clear msb
	shr_u32(  s,m,15);         // build lsb mask
	sub_u32(  m,m,s);          //  from msb
	xor_b32(  r,r,m);          // conditionally invert lsbs
	add_u32(  r,r,s);          // conditionally add 1
	and_b32(  m,r,0x80008000); // extract msb (1 if wrap-around)
	shr_u32(  s,m,15);         // msb ? 1 : 0
	sub_u32(  r,r,s);          // subtract 1 if result wrapped around
	return r;           // halfword-wise absolute value with signed saturation
}

inline unsigned int vadd2(unsigned int a, unsigned int b)
{
	unsigned int s, t;
	s = a ^ b;          // sum bits
	t = a + b;          // actual sum
	s = s ^ t;          // determine carry-ins for each bit position
	s = s & 0x00010000; // carry-in to high word (= carry-out from low word)
	t = t - s;          // subtract out carry-out from low word
	return t;           // halfword-wise sum, with wrap around
}

inline unsigned int vaddss2 (unsigned int a, unsigned int b)
{
	unsigned int r;
	int ahi, alo, blo, bhi, rhi, rlo;
	ahi = (int)((a & 0xffff0000U));
	bhi = (int)((b & 0xffff0000U));
	alo = (int)(a << 16);
	blo = (int)(b << 16);
	add_sat_s32(rlo, alo, blo); 
	add_sat_s32( rhi, ahi, bhi);
	r = ((unsigned int)rhi & 0xffff0000U) | ((unsigned int)rlo >> 16);
	return r;           // halfword-wise sum with signed saturation
}

inline unsigned int vaddus2 (unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int alo, blo, ahi, bhi;
	int rlo, rhi;
	and_b32(     alo, a, 0xffff);    
	and_b32(     blo, b, 0xffff);    
	shr_u32(     ahi, a, 16);        
	shr_u32(     bhi, b, 16);        
	rlo = min ((int)(alo + blo), 65535);
	rhi = min ((int)(ahi + bhi), 65535);
	r = (rhi << 16) + rlo;
	return r;           // halfword-wise sum with unsigned saturation
}

inline unsigned int vavgs2(unsigned int a, unsigned int b)
{
	unsigned int r;
	// avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The 
	// two expressions can be re-written as follows to avoid needing additional
	// intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
	// ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
	// between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
	// ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
	// means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
	// depending on the sign bit of the shifted sum. By handling the msb sum 
	// bit of the result separately, we avoid carry-out during summation and
	// also can use (potentially faster) logical right shifts.
	unsigned int c,s,t,u,v;
	and_b32( u,a,0xfffefffe); // prevent shift crossing chunk boundary
	and_b32( v,b,0xfffefffe); // prevent shift crossing chunk boundary
	xor_b32( s,a,b);          // a ^ b
	and_b32( t,a,b);          // a & b
	shr_u32( u,u,1);          // a >> 1
	shr_u32( v,v,1);          // b >> 1
	and_b32( c,s,0x00010001); // (a ^ b) & 1
	and_b32( s,s,0x80008000); // extract msb (a ^ b)
	and_b32( t,t,0x00010001); // (a & b) & 1
	add_u32( r,u,v);          // (a>>1)+(b>>1) 
	add_u32( r,r,t);          // (a>>1)+(b>>1)+(a&b&1)); rec. msb cy-in
	xor_b32( r,r,s);          // compute msb sum bit: a ^ b ^ cy-in
	shr_u32( t,r,15);         // sign ((a + b) >> 1)
	not_b32( t,t);            // ~sign ((a + b) >> 1)
	and_b32( t,t,c);          // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
	add_u32( r,r,t);          // conditionally add ((a ^ b) & 1)
	return r;           // halfword-wise average of signed integers
}

inline unsigned int vavgu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
	// (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
	c = a ^ b;           
	r = a | b;
	c = c & 0xfffefffe; // ensure shift doesn't cross half-word boundaries
	c = c >> 1;
	r = r - c;
	return r;           // halfword-wise average of unsigned integers
}

inline unsigned int vhaddu2(unsigned int a, unsigned int b)
{
	// HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
	// (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
	unsigned int r, s;
	s = a ^ b;
	r = a & b;
	s = s & 0xfffefffe; // ensure shift doesn't cross halfword boundaries
	s = s >> 1;
	r = r + s;
	return r;           // halfword-wise average of unsigned ints, rounded down
}

inline unsigned int vcmpeq2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x0000 if a == b
	c = r | 0x80008000; // set msbs, to catch carry out
	r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
	c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
	c = r & ~c;         // msb = 1, if r was 0x0000
	shr_u32( r,c,15); // convert
	sub_u32( r,c,r); //  msbs to
	or_b32(  r,c,r); //   mask
	return r;           // halfword-wise (un)signed eq comparison, mask result
}

inline unsigned int vcmpges2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, u;   
	int s2, t2;   

	s2 = (int)((short)(a >> 16));
	t2 = (int)((short)(b >> 16));
	//and_b32(        s,a,0xffff0000); // high word of a
	//      and_b32(        t,b,0xffff0000); // high word of b
	set_ge_s32_s32( u,s2,t2);          // compare two high words
	cvt_s32_s16_s(    s2,a);            // sign-extend low word of a
	cvt_s32_s16_s(    t2,b);            // sign-extend low word of b
	set_ge_s32_s32( s,s2,t2);          // compare two low words
	and_b32(        u,u,0xffff0000); // mask comparison result hi word
	and_b32(        s,s,0x0000ffff); // mask comparison result lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed gt-eq comparison, mask result
}

inline unsigned int vcmpgeu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vavgu2 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
	and_b32( c,c,0x80008000);  // msb = carry-outs
	shr_u32( r,c,15); // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // halfword-wise unsigned gt-eq comparison, mask result
}

inline unsigned int vcmpgts2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, u;   
	int s2, t2;   
	s2 = (int)((short)(a >> 16));
	t2 = (int)((short)(b >> 16));
	//and_b32(        s,a,0xffff0000); // high word of a
	//and_b32(        t,b,0xffff0000); // high word of b
	set_gt_s32_s32( u,s2,t2);          // compare two high words
	cvt_s32_s16_s(    s2,a);            // sign-extend low word of a
	cvt_s32_s16_s(    t2,b);            // sign-extend low word of b
	set_gt_s32_s32( s,s2,t2);          // compare two low words
	and_b32(        u,u,0xffff0000); // mask comparison result hi word
	and_b32(        s,s,0x0000ffff); // mask comparison result lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed gt comparison with mask result
}

inline unsigned int vcmpgtu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vhaddu2 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
	and_b32( c,c,0x80008000);  // msb = carry-outs
	shr_u32( r,c,15); // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // halfword-wise unsigned gt comparison, mask result
}

inline unsigned int vcmples2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, u;   
	int s2, t2;   
	s2 = (int)((short)(a >> 16));
	t2 = (int)((short)(b >> 16));
	//and_b32(        s,a,0xffff0000); // high word of a
	//and_b32(        t,b,0xffff0000); // high word of b
	set_le_s32_s32( u,s2,t2);          // compare two high words
	cvt_s32_s16_s(    s2,a);            // sign-extend low word of a
	cvt_s32_s16_s(    t2,b);            // sign-extend low word of b
	set_le_s32_s32( s,s2,t2);          // compare two low words
	and_b32(        u,u,0xffff0000); // mask comparison result hi word
	and_b32(        s,s,0x0000ffff); // mask comparison result lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed lt-eq comparison, mask result
}

inline unsigned int vcmpleu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vavgu2 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
	and_b32( c,c,0x80008000);  // msb = carry-outs
	shr_u32( r,c,15); // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // halfword-wise unsigned lt-eq comparison, mask result
}

inline unsigned int vcmplts2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, u;   
	int s2, t2;   
	s2 = (int)((short)(a >> 16));
	t2 = (int)((short)(b >> 16));
	//and_b32(        s,a,0xffff0000); // high word of a
	//and_b32(        t,b,0xffff0000); // high word of b
	set_lt_s32_s32( u,s2,t2);          // compare two high words
	cvt_s32_s16_s(    s2,a);            // sign-extend low word of a
	cvt_s32_s16_s(    t2,b);            // sign-extend low word of b
	set_lt_s32_s32( s,s2,t2);          // compare two low words
	and_b32(        u,u,0xffff0000); // mask comparison result hi word
	and_b32(        s,s,0x0000ffff); // mask comparison result lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed lt comparison with mask result
}

inline unsigned int vcmpltu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vhaddu2 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
	and_b32( c,c,0x80008000);  // msb = carry-outs
	shr_u32( r,c,15); // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // halfword-wise unsigned lt comparison, mask result
}

inline unsigned int vcmpne2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x0000 if a == b
	c = r | 0x80008000; // set msbs, to catch carry out
	c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
	c = r | c;          // msb = 1, if r was not 0x0000
	and_b32( c,c,0x80008000);  // extract msbs
	shr_u32( r,c,15); // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // halfword-wise (un)signed ne comparison, mask result
}

inline unsigned int vabsdiffu2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u, v;
	s = a & 0x0000ffff; // extract low halfword
	r = b & 0x0000ffff; // extract low halfword
	u = max (r, s);     // maximum of low halfwords
	v = min (r, s);     // minimum of low halfwords
	s = a & 0xffff0000; // extract high halfword
	r = b & 0xffff0000; // extract high halfword
	t = max (r, s);     // maximum of high halfwords
	s = min (r, s);     // minimum of high halfwords
	r = u | t;          // maximum of both halfwords
	s = v | s;          // minimum of both halfwords
	r = r - s;          // |a - b| = max(a,b) - min(a,b);
	return r;           // halfword-wise absolute difference of unsigned ints
}

inline unsigned int vmaxs2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u;
	cvt_s32_s16_s( r,a); // extract low halfword
	cvt_s32_s16_s( s,b); // extract low halfword
	t = max((int)r,(int)s); // maximum of low halfwords
	r = a & 0xffff0000;     // extract high halfword
	s = b & 0xffff0000;     // extract high halfword
	u = max((int)r,(int)s); // maximum of high halfwords
	r = u | (t & 0xffff);   // combine halfword maximums
	return r;           // halfword-wise maximum of signed integers
}

inline unsigned int vmaxu2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u;
	r = a & 0x0000ffff; // extract low halfword
	s = b & 0x0000ffff; // extract low halfword
	t = max (r, s);     // maximum of low halfwords
	r = a & 0xffff0000; // extract high halfword
	s = b & 0xffff0000; // extract high halfword
	u = max (r, s);     // maximum of high halfwords
	r = t | u;          // combine halfword maximums
	return r;           // halfword-wise maximum of unsigned integers
}

inline unsigned int vmins2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u;
	cvt_s32_s16_s( r,a); // extract low halfword
	cvt_s32_s16_s( s,b); // extract low halfword
	t = min((int)r,(int)s); // minimum of low halfwords
	r = a & 0xffff0000;     // extract high halfword
	s = b & 0xffff0000;     // extract high halfword
	u = min((int)r,(int)s); // minimum of high halfwords
	r = u | (t & 0xffff);   // combine halfword minimums
	return r;           // halfword-wise minimum of signed integers
}

inline unsigned int vminu2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u;
	r = a & 0x0000ffff; // extract low halfword
	s = b & 0x0000ffff; // extract low halfword
	t = min (r, s);     // minimum of low halfwords
	r = a & 0xffff0000; // extract high halfword
	s = b & 0xffff0000; // extract high halfword
	u = min (r, s);     // minimum of high halfwords
	r = t | u;          // combine halfword minimums
	return r;           // halfword-wise minimum of unsigned integers
}

inline unsigned int vseteq2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x0000 if a == b
	c = r | 0x80008000; // set msbs, to catch carry out
	r = r ^ c;          // extract msbs, msb = 1 if r < 0x8000
	c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
	c = r & ~c;         // msb = 1, if r was 0x0000
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise (un)signed eq comparison, bool result
}

// per-halfword signed comparison: a >= b ? 1 : 0
inline unsigned int vsetges2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;   
	and_b32(        s,a,0xffff0000); // high word of a
	and_b32(        t,b,0xffff0000); // high word of b
	set_ge_s32_s32( u,(int)s,(int)t);          // compare two high words
	cvt_s32_s16_u(    s,a);            // sign-extend low word of a
	cvt_s32_s16_u(    t,b);            // sign-extend low word of b
	set_ge_s32_s32( s,(int)s,(int)t);          // compare two low words
	and_b32(        u,u,0x00010000); // extract bool result of hi word
	and_b32(        s,s,0x00000001); // extract bool result of lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed gt-eq comparison, bool result
}

inline unsigned int vsetgeu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vavgu2 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
	c = c & 0x80008000; // msb = carry-outs
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise unsigned gt-eq comparison, bool result
}

inline unsigned int vsetgts2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;   
	and_b32(        s,a,0xffff0000); // high word of a
	and_b32(        t,b,0xffff0000); // high word of b
	set_gt_s32_s32( u,s,t);          // compare two high words
	cvt_s32_s16_u(    s,a);            // sign-extend low word of a
	cvt_s32_s16_u(    t,b);            // sign-extend low word of b
	set_gt_s32_s32( s,(int)s,(int)t);          // compare two low words
	and_b32(        u,u,0x00010000); // extract bool result of hi word
	and_b32(        s,s,0x00000001); // extract bool result of lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed gt comparison with bool result
}

inline unsigned int vsetgtu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vhaddu2 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
	c = c & 0x80008000; // msbs = carry-outs
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise unsigned gt comparison, bool result
}

inline unsigned int vsetles2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;   
	and_b32(        s,a,0xffff0000); // high word of a
	and_b32(        t,b,0xffff0000); // high word of b
	set_le_s32_s32( u,s,t);          // compare two high words
	cvt_s32_s16_u(    s,a);            // sign-extend low word of a
	cvt_s32_s16_u(    t,b);            // sign-extend low word of b
	set_le_s32_s32( s,(int)s,(int)t);          // compare two low words
	and_b32(        u,u,0x00010000); // extract bool result of hi word
	and_b32(        s,s,0x00000001); // extract bool result of lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed lt-eq comparison, bool result
}

inline unsigned int vsetleu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vavgu2 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
	c = c & 0x80008000; // msb = carry-outs
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise unsigned lt-eq comparison, bool result
}

inline unsigned int vsetlts2(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;   
	and_b32(        s,a,0xffff0000); // high word of a
	and_b32(        t,b,0xffff0000); // high word of b
	set_lt_s32_s32( u,s,t);          // compare two high words
	cvt_s32_s16_u(    s,a);            // sign-extend low word of a
	cvt_s32_s16_u(    t,b);            // sign-extend low word of b
	set_lt_s32_s32( s,(int)s,(int)t);          // compare two low words
	and_b32(        u,u,0x00010000); // extract bool result of hi word
	and_b32(        s,s,0x00000001); // extract bool result of lo word
	or_b32(         r,s,u);          // combine the two results
	return r;           // halfword-wise signed lt comparison with bool result
}

inline unsigned int vsetltu2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vhaddu2 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
	c = c & 0x80008000; // msb = carry-outs
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise unsigned lt comparison, bool result
}

inline unsigned int vsetne2(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x0000 if a == b
	c = r | 0x80008000; // set msbs, to catch carry out
	c = c - 0x00010001; // msb = 0, if r was 0x0000 or 0x8000
	c = r | c;          // msb = 1, if r was not 0x0000
	c = c & 0x80008000; // extract msbs
	r = c >> 15;        // convert to bool
	return r;           // halfword-wise (un)signed ne comparison, bool result
}

inline unsigned int vsadu2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	unsigned int t, u, v;
	s = a & 0x0000ffff; // extract low halfword
	r = b & 0x0000ffff; // extract low halfword
	u = max (r, s);     // maximum of low halfwords
	v = min (r, s);     // minimum of low halfwords
	s = a & 0xffff0000; // extract high halfword
	r = b & 0xffff0000; // extract high halfword
	t = max (r, s);     // maximum of high halfwords
	s = min (r, s);     // minimum of high halfwords
	u = u - v;          // low halfword: |a - b| = max(a,b) - min(a,b); 
	t = t - s;          // high halfword: |a - b| = max(a,b) - min(a,b);
	shr_u32( t,t,16);
	r = t + u;          // sum absolute halfword differences
	return r;           // halfword-wise sum of abs differences of unsigned int
}

inline unsigned int vsub2(unsigned int a, unsigned int b)
{
	unsigned int s, t;
	s = a ^ b;          // sum bits
	t = a - b;          // actual sum
	s = s ^ t;          // determine carry-ins for each bit position
	s = s & 0x00010000; // borrow to high word 
	t = t + s;          // compensate for borrow from low word
	return t;           // halfword-wise difference
}

inline unsigned int vsubss2 (unsigned int a, unsigned int b)
{
	unsigned int r;
	int ahi, alo, blo, bhi, rhi, rlo;
	ahi = (int)((a & 0xffff0000U));
	bhi = (int)((b & 0xffff0000U));
	alo = (int)(a << 16);
	blo = (int)(b << 16);
	sub_sat_s32( rlo, alo, blo);
	sub_sat_s32( rhi, ahi, bhi);
	r = ((unsigned int)rhi & 0xffff0000U) | ((unsigned int)rlo >> 16);
	return r;           // halfword-wise difference with signed saturation
}

// per-halfword subtraction w/ unsigned saturation: sat.u16(a-b)
inline unsigned int vsubus2 (unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int alo, blo, ahi, bhi;
	int rlo, rhi;
	and_b32(     alo, a, 0xffff);    
	and_b32(     blo, b, 0xffff);    
	shr_u32(     ahi, a, 16);        
	shr_u32(     bhi, b, 16);        
	rlo = max ((int)((unsigned short)alo) - (int)((unsigned short)blo), 0);
	rhi = max ((int)((unsigned short)ahi) - (int)((unsigned short)bhi), 0);

	r = rhi * 65536 + rlo;
	return r;           // halfword-wise difference with unsigned saturation
}

inline unsigned int vneg2(unsigned int a)
{
	return vsub2 (0, a);// halfword-wise negation with wrap-around
}

inline unsigned int vnegss2(unsigned int a)
{
	return vsubss2(0,a);// halfword-wise negation with signed saturation
}

inline unsigned int vabsdiffs2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpges2 (a, b);// mask = 0xff if a >= b
	r = a ^ b;          //
	s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
	r = s ^ r;          // select a when b >= a, else select b => min(a,b)
	r = vsub2 (s, r);   // |a - b| = max(a,b) - min(a,b);
	return r;           // halfword-wise absolute difference of signed integers
}

inline unsigned int vsads2(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vabsdiffs2 (a, b);
	r = (s >> 16) + (s & 0x0000ffff);
	return r;           // halfword-wise sum of abs. differences of signed ints
}

inline unsigned int vabs4(unsigned int a)
{
	unsigned int r;
	unsigned int m,s; 
	and_b32(  m,a,0x80808080); // extract msb
	and_b32(  r,a,0x7f7f7f7f); // clear msb
	shr_u32(  s,m,7);          // build lsb mask
	sub_u32(  m,m,s);          //  from msb
	xor_b32(  r,r,m);          // conditionally invert lsbs
	add_u32(  r,r,s);          // conditionally add 1
	return r;           // byte-wise absolute value, with wrap-around
}

inline unsigned int vabsss4(unsigned int a)
{
	unsigned int r;
    
	unsigned int m,s;      
	and_b32(  m,a,0x80808080); // extract msb
	and_b32(  r,a,0x7f7f7f7f); // clear msb
	shr_u32(  s,m,7);          // build lsb mask
	sub_u32(  m,m,s);          //  from msb
	xor_b32(  r,r,m);          // conditionally invert lsbs
	add_u32(  r,r,s);          // conditionally add 1
	and_b32(  m,r,0x80808080); // extract msb (1 if wrap-around)
	shr_u32(  s,m,7);          // msb ? 1 : 0
	sub_u32(  r,r,s);          // subtract 1 if result wrapped around
	return r;           // byte-wise absolute value with signed saturation
}

inline unsigned int vadd4(unsigned int a, unsigned int b)
{
	unsigned int r, s, t;
	s = a ^ b;          // sum bits
	r = a & 0x7f7f7f7f; // clear msbs
	t = b & 0x7f7f7f7f; // clear msbs
	s = s & 0x80808080; // msb sum bits
	r = r + t;          // add without msbs, record carry-out in msbs
	r = r ^ s;          // sum of msb sum and carry-in bits, w/o carry-out
	return r;           // byte-wise sum, with wrap-around
}

inline unsigned int vaddss4 (unsigned int a, unsigned int b)
{
	/*
	For signed saturation, saturation is controlled by the overflow signal: 
	ovfl = (carry-in to msb) XOR (carry-out from msb). Overflow can only 
	occur when the msbs of both inputs are the same. The defined response to
	overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
	and 0x80 when the addends are negative (bit 7 set). The truth table for
	the msb is

	a   b   cy_in  res  cy_out  ovfl
	--------------------------------
	0   0       0    0       0     0
	0   0       1    1       0     1
	0   1       0    1       0     0
	0   1       1    0       1     0
	1   0       0    1       0     0
	1   0       1    0       1     0
	1   1       0    0       1     1
	1   1       1    1       1     0

	The seven low-order bits can be handled by simple wrapping addition with
	the carry out from bit 6 recorded in the msb (thus corresponding to the 
	cy_in in the truth table for the msb above). ovfl can be computed in many
	equivalent ways, here we use ovfl = (a ^ carry_in) & ~(a ^ b) since we 
	already need to compute (a ^ b) for the msb sum bit computation. First we
	compute the normal, wrapped addition result. When overflow is detected,
	we mask off the msb of the result, then compute a mask covering the seven
	low order bits, which are all set to 1. This sets the byte to 0x7f as we
	previously cleared the msb. In the overflow case, the sign of the result
	matches the sign of either of the inputs, so we extract the sign of a and
	add it to the low order bits, which turns 0x7f into 0x80, the correct 
	result for an overflowed negative result.
	*/
	unsigned int r;
	unsigned int s,t,u;    
	and_b32(  r, a, 0x7f7f7f7f); // clear msbs
	and_b32(  t, b, 0x7f7f7f7f); // clear msbs
	xor_b32(  s, a, b);          // sum bits = (a ^ b)
	add_u32(  r, r, t);          // capture msb carry-in in bit 7
	xor_b32(  t, a, r);          // a ^ carry_in
	not_b32(  u, s);             // ~(a ^ b)
	and_b32(  t, t, u);          // ovfl = (a ^ carry_in) & ~(a ^ b)
	and_b32(  s, s, 0x80808080); // msb sum bits
	xor_b32(  r, r, s);          // msb result = (a ^ b ^ carry_in)
	and_b32(  t, t, 0x80808080); // ovfl ? 0x80 : 0
	shr_u32(  s, t, 7);          // ovfl ? 1 : 0
	not_b32(  u, t);             // ovfl ? 0x7f : 0xff
	and_b32(  r, r, u);          // ovfl ? (a + b) & 0x7f : a + b
	and_b32(  u, a, t);          // ovfl ? a & 0x80 : 0
	sub_u32(  t, t, s);          // ovfl ? 0x7f : 0
	shr_u32(  u, u, 7);          // ovfl ? sign(a) : 0
	or_b32(   r, r, t);          // ovfl ? 0x7f : a + b
	add_u32(  r, r, u);          // ovfl ? 0x7f+sign(a) : a + b
	return r;           // byte-wise sum with signed saturation
}

inline unsigned int vaddus4 (unsigned int a, unsigned int b)
{
	// This code uses the same basic approach used for non-saturating addition.
	// The seven low-order bits in each byte are summed by regular addition,
	// with the carry-out from bit 6 (= carry-in for the msb) being recorded 
	// in bit 7, while the msb is handled separately.
	//
	// The fact that this is a saturating addition simplifies the handling of
	// the msb. When carry-out from the msb occurs, the entire byte must be
	// written as 0xff, and the computed msb is overwritten in the process. 
	// The corresponding entries in the truth table for the result msb thus 
	// become "don't cares":
	//
	// a  b  cy-in  res  cy-out
	// ------------------------
	// 0  0    0     0     0
	// 0  0    1     1     0
	// 0  1    0     1     0
	// 0  1    1     X     1
	// 1  0    0     1     0
	// 1  0    1     X     1
	// 1  1    0     X     1
	// 1  1    1     X     1
	//
	// As is easily seen, the simplest implementation of the result msb bit is 
	// simply (a | b | cy-in), with masking needed to isolate the msb. Note 
	// that this computation also makes the msb handling redundant with the 
	// clamping to 0xFF, because the msb is already set to 1 when saturation 
	// occurs. This means we only need to apply saturation to the seven lsb
	// bits in each byte, by overwriting with 0x7F. Saturation is controlled
	// by carry-out from the msb, which can be represented by various Boolean
	// expressions. Since to compute (a | b | cy-in) we need to compute (a | b)
	// anyhow, most efficient of these is cy-out = ((a & b) | cy-in) & (a | b).
	unsigned int r;
	unsigned int s,t,m;    
	or_b32(   m, a, b);          // (a | b)
	and_b32(  r, a, 0x7f7f7f7f); // clear msbs
	and_b32(  t, b, 0x7f7f7f7f); // clear msbs
	and_b32(  m, m, 0x80808080); // (a | b), isolate msbs
	add_u32(  r, r, t);          // add w/o msbs, record msb-carry-ins
	and_b32(  t, a, b);          // (a & b)
	or_b32(   t, t, r);          // (a & b) | cy-in)
	or_b32(   r, r, m);          // msb = cy-in | (a | b)
	and_b32(  t, t, m);          // cy-out=((a&b)|cy-in)&(a|b),in msbs
	shr_u32(  s, t, 7);          // cy-out ? 1 : 0
	sub_u32(  t, t, s);          // lsb-overwrite: cy-out ? 0x7F : 0
	or_b32(   r, r, t);          // conditionally overwrite lsbs
	return r;           // byte-wise sum with unsigned saturation
}

inline unsigned int vavgs4(unsigned int a, unsigned int b)
{
	unsigned int r;
	// avgs (a + b) = ((a + b) < 0) ? ((a + b) >> 1) : ((a + b + 1) >> 1). The 
	// two expressions can be re-written as follows to avoid needing additional
	// intermediate bits: ((a + b) >> 1) = (a >> 1) + (b >> 1) + ((a & b) & 1),
	// ((a + b + 1) >> 1) = (a >> 1) + (b >> 1) + ((a | b) & 1). The difference
	// between the two is ((a ^ b) & 1). Note that if (a + b) < 0, then also
	// ((a + b) >> 1) < 0, since right shift rounds to negative infinity. This
	// means we can compute ((a + b) >> 1) then conditionally add ((a ^ b) & 1)
	// depending on the sign bit of the shifted sum. By handling the msb sum 
	// bit of the result separately, we avoid carry-out during summation and
	// also can use (potentially faster) logical right shifts.
	unsigned int c,s,t,u,v;
	and_b32( u,a,0xfefefefe); // prevent shift crossing chunk boundary
	and_b32( v,b,0xfefefefe); // prevent shift crossing chunk boundary
	xor_b32( s,a,b);          // a ^ b
	and_b32( t,a,b);          // a & b
	shr_u32( u,u,1);          // a >> 1
	shr_u32( v,v,1);          // b >> 1
	and_b32( c,s,0x01010101); // (a ^ b) & 1
	and_b32( s,s,0x80808080); // extract msb (a ^ b)
	and_b32( t,t,0x01010101); // (a & b) & 1
	add_u32( r,u,v);          // (a>>1)+(b>>1) 
	add_u32( r,r,t);          // (a>>1)+(b>>1)+(a&b&1)); rec. msb cy-in
	xor_b32( r,r,s);          // compute msb sum bit: a ^ b ^ cy-in
	shr_u32( t,r,7);          // sign ((a + b) >> 1)
	not_b32( t,t);            // ~sign ((a + b) >> 1)
	and_b32( t,t,c);          // ((a ^ b) & 1) & ~sign ((a + b) >> 1)
	add_u32( r,r,t);          // conditionally add ((a ^ b) & 1)
	return r;           // byte-wise average of signed integers
}

inline unsigned int vavgu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// HAKMEM #23: a + b = 2 * (a | b) - (a ^ b) ==>
	// (a + b + 1) / 2 = (a | b) - ((a ^ b) >> 1)
	c = a ^ b;           
	r = a | b;
	c = c & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
	c = c >> 1;
	r = r - c;
	return r;           // byte-wise average of unsigned integers
}

inline unsigned int vhaddu4(unsigned int a, unsigned int b)
{
	// HAKMEM #23: a + b = 2 * (a & b) + (a ^ b) ==>
	// (a + b) / 2 = (a & b) + ((a ^ b) >> 1)
	unsigned int r, s;   
	s = a ^ b;           
	r = a & b;
	s = s & 0xfefefefe; // ensure following shift doesn't cross byte boundaries
	s = s >> 1;
	s = r + s;
	return s;           // byte-wise average of unsigned integers, rounded down
}

inline unsigned int vcmpeq4(unsigned int a, unsigned int b)
{
	unsigned int c, r;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x00 if a == b
	c = r | 0x80808080; // set msbs, to catch carry out
	r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
	c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
	c = r & ~c;         // msb = 1, if r was 0x00
	shr_u32( r,c,7);  // convert
	sub_u32( r,c,r); //  msbs to
	or_b32(  r,c,r); //   mask
	return r;           // byte-wise (un)signed eq comparison with mask result
}

inline unsigned int vcmpges4(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t;
	xor_b32(     s,a,b);          // a ^ b
	or_b32(      r,a,0x80808080); // set msbs
	and_b32(     t,b,0x7f7f7f7f); // clear msbs
	sub_u32(     r,r,t);          // subtract lsbs, msb: ~borrow-in
	xor_b32(     t,r,a);          // msb: ~borrow-in ^ a
	xor_b32(     r,r,s);          // msb: ~sign(res) = a^b^~borrow-in
	and_b32(     t,t,s);          // msb: ovfl= (~bw-in ^ a) & (a ^ b)
	xor_b32(     t,t,r);          // msb: ge = ovfl != ~sign(res)
	and_b32(     t,t,0x80808080); // isolate msbs = ovfl
	shr_u32(     r,t,7);          // build mask
	sub_u32(     r,t,r);          //  from
	or_b32(      r,r,t);          //   msbs
	return r;           // byte-wise signed gt-eq comparison with mask result
}

inline unsigned int vcmpgeu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vavgu4 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
	and_b32( c,c,0x80808080);  // msb = carry-outs
	shr_u32( r,c,7);  // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // byte-wise unsigned gt-eq comparison with mask result
}

inline unsigned int vcmpgts4(unsigned int a, unsigned int b)
{
	unsigned int r;
	/* a <= b <===> a + ~b < 0 */
     
	unsigned int s,t,u;  
	not_b32(  b,b);           
	and_b32(  r,a,0x7f7f7f7f); // clear msbs
	and_b32(  t,b,0x7f7f7f7f); // clear msbs
	xor_b32(  s,a,b);          // sum bits = (a ^ b)
	add_u32(  r,r,t);          // capture msb carry-in in bit 7
	xor_b32(  t,a,r);          // a ^ carry_in
	not_b32(  u,s);            // ~(a ^ b)
	and_b32(  t,t,u);          // msb: ovfl = (a ^ carry_in) & ~(a^b)
	xor_b32(  r,r,u);          // msb: ~result = (~(a ^ b) ^ carry_in)
	xor_b32(  t,t,r);          // msb: gt = ovfl != sign(~res)
	and_b32(  t,t,0x80808080); // isolate msbs
	shr_u32(  r,t,7);          // build mask
	sub_u32(  r,t,r);          //  from
	or_b32(   r,r,t);          //   msbs
	return r;           // byte-wise signed gt comparison with mask result
}

inline unsigned int vcmpgtu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vhaddu4 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
	and_b32( c,c,0x80808080);  // msb = carry-outs
	shr_u32( r,c,7);  // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // byte-wise unsigned gt comparison with mask result
}

inline unsigned int vcmples4(unsigned int a, unsigned int b)
{
	unsigned int r;
	/* a <= b <===> a + ~b < 0 */
     
	unsigned int s,t,u;  
	not_b32(  u,b);            // ~b
	and_b32(  r,a,0x7f7f7f7f); // clear msbs
	and_b32(  t,u,0x7f7f7f7f); // clear msbs
	xor_b32(  u,a,b);          // sum bits = (a ^ b)
	add_u32(  r,r,t);          // capture msb carry-in in bit 7
	xor_b32(  t,a,r);          // a ^ carry_in
	not_b32(  s,u);            // ~(a ^ b)
	and_b32(  t,t,u);          // msb: ovfl = (a ^ carry_in) & (a ^ b)
	xor_b32(  r,r,s);          // msb: result = (a ^ ~b ^ carry_in)
	xor_b32(  t,t,r);          // msb: le = ovfl != sign(res)
	and_b32(  t,t,0x80808080); // isolate msbs
	shr_u32(  r,t,7);          // build mask
	sub_u32(  r,t,r);          //  from
	or_b32(   r,r,t);          //   msbs
	return r;           // byte-wise signed lt-eq comparison with mask result
}

inline unsigned int vcmpleu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vavgu4 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
	and_b32( c,c,0x80808080);  // msb = carry-outs
	shr_u32( r,c,7);  // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // byte-wise unsigned lt-eq comparison with mask result
}

inline unsigned int vcmplts4(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;
	not_b32(     u,b);            // ~b
	xor_b32(     s,u,a);          // a ^ ~b
	or_b32(      r,a,0x80808080); // set msbs
	and_b32(     t,b,0x7f7f7f7f); // clear msbs
	sub_u32(     r,r,t);          // subtract lsbs, msb: ~borrow-in
	xor_b32(     t,r,a);          // msb: ~borrow-in ^ a
	not_b32(     u,s);            // msb: ~(a^~b)
	xor_b32(     r,r,s);          // msb: res = a ^ ~b ^ ~borrow-in
	and_b32(     t,t,u);          // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
	xor_b32(     t,t,r);          // msb: lt = ovfl != sign(res)
	and_b32(     t,t,0x80808080); // isolate msbs
	shr_u32(     r,t,7);          // build mask
	sub_u32(     r,t,r);          //  from
	or_b32(      r,r,t);          //   msbs
	return r;           // byte-wise signed lt comparison with mask result
}

inline unsigned int vcmpltu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vhaddu4 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
	and_b32( c,c,0x80808080);  // msb = carry-outs
	shr_u32( r,c,7);  // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // byte-wise unsigned lt comparison with mask result
}

inline unsigned int vcmpne4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x00 if a == b
	c = r | 0x80808080; // set msbs, to catch carry out
	c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
	c = r | c;          // msb = 1, if r was not 0x00
	and_b32( c,c,0x80808080);  // extract msbs
	shr_u32( r,c,7);  // build mask
	sub_u32( r,c,r); //  from
	or_b32(  r,c,r); //   msbs
	return r;           // byte-wise (un)signed ne comparison with mask result
}

inline unsigned int vabsdiffu4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpgeu4 (a, b);// mask = 0xff if a >= b
	r = a ^ b;          //
	s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
	r = s ^ r;          // select a when b >= a, else select b => min(a,b)
	r = s - r;          // |a - b| = max(a,b) - min(a,b);
	return r;           // byte-wise absolute difference of unsigned integers
}

inline unsigned int vmaxs4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpges4 (a, b);// mask = 0xff if a >= b
	r = a & s;          // select a when b >= a
	s = b & ~s;         // select b when b < a
	r = r | s;          // combine byte selections
	return r;           // byte-wise maximum of signed integers
}

inline unsigned int vmaxu4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpgeu4 (a, b);// mask = 0xff if a >= b
	r = a & s;          // select a when b >= a
	s = b & ~s;         // select b when b < a
	r = r | s;          // combine byte selections
	return r;           // byte-wise maximum of unsigned integers
}

inline unsigned int vmins4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpges4 (b, a);// mask = 0xff if a >= b
	r = a & s;          // select a when b >= a
	s = b & ~s;         // select b when b < a
	r = r | s;          // combine byte selections
	return r;           // byte-wise minimum of signed integers
}

inline unsigned int vminu4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpgeu4 (b, a);// mask = 0xff if a >= b
	r = a & s;          // select a when b >= a
	s = b & ~s;         // select b when b < a
	r = r | s;          // combine byte selections
	return r;           // byte-wise minimum of unsigned integers
}
inline unsigned int vseteq4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x00 if a == b
	c = r | 0x80808080; // set msbs, to catch carry out
	r = r ^ c;          // extract msbs, msb = 1 if r < 0x80
	c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
	c = r & ~c;         // msb = 1, if r was 0x00
	r = c >> 7;         // convert to bool
	return r;           // byte-wise (un)signed eq comparison with bool result
}

inline unsigned int vsetles4(unsigned int a, unsigned int b)
{
	unsigned int r;
	/* a <= b <===> a + ~b < 0 */
     
	unsigned int s,t,u;  
	not_b32(  u,b);            // ~b
	and_b32(  r,a,0x7f7f7f7f); // clear msbs
	and_b32(  t,u,0x7f7f7f7f); // clear msbs
	xor_b32(  u,a,b);          // sum bits = (a ^ b)
	add_u32(  r,r,t);          // capture msb carry-in in bit 7
	xor_b32(  t,a,r);          // a ^ carry_in
	not_b32(  s,u);            // ~(a ^ b)
	and_b32(  t,t,u);          // msb: ovfl = (a ^ carry_in) & (a ^ b)
	xor_b32(  r,r,s);          // msb: result = (a ^ ~b ^ carry_in)
	xor_b32(  t,t,r);          // msb: le = ovfl != sign(res)
	and_b32(  t,t,0x80808080); // isolate msbs
	shr_u32(  r,t,7);          // convert to bool
	return r;           // byte-wise signed lt-eq comparison with bool result
}

inline unsigned int vsetleu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vavgu4 (a, b);  // (b + ~a + 1) / 2 = (b - a) / 2
	c = c & 0x80808080; // msb = carry-outs
	r = c >> 7;         // convert to bool
	return r;           // byte-wise unsigned lt-eq comparison with bool result
}

inline unsigned int vsetlts4(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t, u;
	not_b32(     u,b);            // ~b
	or_b32(      r,a,0x80808080); // set msbs
	and_b32(     t,b,0x7f7f7f7f); // clear msbs
	xor_b32(     s,u,a);          // a ^ ~b
	sub_u32(     r,r,t);          // subtract lsbs, msb: ~borrow-in
	xor_b32(     t,r,a);          // msb: ~borrow-in ^ a
	not_b32(     u,s);            // msb: ~(a^~b)
	xor_b32(     r,r,s);          // msb: res = a ^ ~b ^ ~borrow-in
	and_b32(     t,t,u);          // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
	xor_b32(     t,t,r);          // msb: lt = ovfl != sign(res)
	and_b32(     t,t,0x80808080); // isolate msbs
	shr_u32(     r,t,7);          // convert to bool
	return r;           // byte-wise signed lt comparison with bool result
}

inline unsigned int vsetltu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( a,a);
	c = vhaddu4 (a, b); // (b + ~a) / 2 = (b - a) / 2 [rounded down]
	c = c & 0x80808080; // msb = carry-outs
	r = c >> 7;         // convert to bool
	return r;           // byte-wise unsigned lt comparison with bool result
}

inline unsigned int vsetges4(unsigned int a, unsigned int b)
{
	unsigned int r;
	unsigned int s, t;
	xor_b32(     s,a,b);          // a ^ b
	or_b32(      r,a,0x80808080); // set msbs
	and_b32(     t,b,0x7f7f7f7f); // clear msbs
	sub_u32(     r,r,t);          // subtract lsbs, msb: ~borrow-in
	xor_b32(     t,r,a);          // msb: ~borrow-in ^ a
	xor_b32(     r,r,s);          // msb: ~sign(res) = a^b^~borrow-in
	and_b32(     t,t,s);          // msb: ovfl= (~bw-in ^ a) & (a ^ b)
	xor_b32(     t,t,r);          // msb: ge = ovfl != ~sign(res)
	and_b32(     t,t,0x80808080); // isolate msbs
	shr_u32(     r,t,7);          // convert to bool
	return r;           // byte-wise signed gt-eq comparison with bool result
}

inline unsigned int vsetgeu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vavgu4 (a, b);  // (a + ~b + 1) / 2 = (a - b) / 2
	c = c & 0x80808080; // msb = carry-outs
	r = c >> 7;         // convert to bool
	return r;           // byte-wise unsigned gt-eq comparison with bool result
}

inline unsigned int vsetgts4(unsigned int a, unsigned int b)
{
	unsigned int r;
	/* a <= b <===> a + ~b < 0 */
     
	unsigned int s,t,u;  
	not_b32(  b,b);           
	and_b32(  r,a,0x7f7f7f7f); // clear msbs
	and_b32(  t,b,0x7f7f7f7f); // clear msbs
	xor_b32(  s,a,b);          // sum bits = (a ^ b)
	add_u32(  r,r,t);          // capture msb carry-in in bit 7
	xor_b32(  t,a,r);          // a ^ carry_in
	not_b32(  u,s);            // ~(a ^ b)
	and_b32(  t,t,u);          // msb: ovfl = (a ^ carry_in) & ~(a^b)
	xor_b32(  r,r,u);          // msb: ~result = (~(a ^ b) ^ carry_in)
	xor_b32(  t,t,r);          // msb: gt = ovfl != sign(~res)
	and_b32(  t,t,0x80808080); // isolate msbs
	shr_u32(  r,t,7);          // convert to bool
	return r;           // byte-wise signed gt comparison with mask result
}

inline unsigned int vsetgtu4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	not_b32( b,b);
	c = vhaddu4 (a, b); // (a + ~b) / 2 = (a - b) / 2 [rounded down]
	c = c & 0x80808080; // msb = carry-outs
	r = c >> 7;         // convert to bool
	return r;           // byte-wise unsigned gt comparison with bool result
}

inline unsigned int vsetne4(unsigned int a, unsigned int b)
{
	unsigned int r, c;
	// inspired by Alan Mycroft's null-byte detection algorithm:
	// null_byte(x) = ((x - 0x01010101) & (~x & 0x80808080))
	r = a ^ b;          // 0x00 if a == b
	c = r | 0x80808080; // set msbs, to catch carry out
	c = c - 0x01010101; // msb = 0, if r was 0x00 or 0x80
	c = r | c;          // msb = 1, if r was not 0x00
	c = c & 0x80808080; // extract msbs
	r = c >> 7;         // convert to bool
	return r;           // byte-wise (un)signed ne comparison with bool result
}

inline unsigned int vsadu4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	r = vabsdiffu4 (a, b);
	s = r >> 8;
	r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
	r = ((r << 16) + r) >> 16;
	return r;           // byte-wise sum of absol. differences of unsigned ints
}

inline unsigned int vsub4(unsigned int a, unsigned int b)
{
	unsigned int r, s, t;
	s = a ^ ~b;         // inverted sum bits
	r = a | 0x80808080; // set msbs
	t = b & 0x7f7f7f7f; // clear msbs
	s = s & 0x80808080; // inverted msb sum bits
	r = r - t;          // subtract w/o msbs, record inverted borrows in msb
	r = r ^ s;          // combine inverted msb sum bits and borrows
	return r;           // byte-wise difference
}

inline unsigned int vsubss4(unsigned int a, unsigned int b)
{
	unsigned int r;
	/*
	For signed saturation, saturation is controlled by the overflow signal: 
	ovfl = (borrow-in to msb) XOR (borrow-out from msb). Overflow can only 
	occur when the msbs of both inputs are differemt. The defined response to
	overflow is to deliver 0x7f when the addends are positive (bit 7 clear),
	and 0x80 when the addends are negative (bit 7 set). The truth table for
	the msb is

	a   b  bw_in  res  bw_out  ovfl  a^~bw_in  ~(a^~b) (a^~bw_in)&~(a^~b)
	---------------------------------------------------------------------
	0   0      0    0       0     0         1        0                  0
	0   0      1    1       1     0         0        0                  0
	0   1      0    1       1     1         1        1                  1
	0   1      1    0       1     0         0        1                  0
	1   0      0    1       0     0         0        1                  0
	1   0      1    0       0     1         1        1                  1
	1   1      0    0       0     0         0        0                  0
	1   1      1    1       1     0         1        0                  0

	The seven low-order bits can be handled by wrapping subtraction with the
	borrow-out from bit 6 recorded in the msb (thus corresponding to the 
	bw_in in the truth table for the msb above). ovfl can be computed in many
	equivalent ways, here we use ovfl = (a ^ ~borrow_in) & ~(a ^~b) since we 
	already need to compute (a ^~b) and ~borrow-in for the msb result bit 
	computation. First we compute the normal, wrapped subtraction result. 
	When overflow is detected, we mask off the result's msb, then compute a
	mask covering the seven low order bits, which are all set to 1. This sets
	the byte to 0x7f as we previously cleared the msb. In the overflow case, 
	the sign of the result matches the sign of input a, so we extract the 
	sign of a and add it to the low order bits, which turns 0x7f into 0x80, 
	the correct result for an overflowed negative result.
	*/

	unsigned int s,t,u; 
	not_b32(     u,b);            // ~b
	xor_b32(     s,u,a);          // a ^ ~b
	or_b32(      r,a,0x80808080); // set msbs
	and_b32(     t,b,0x7f7f7f7f); // clear msbs
	sub_u32(     r,r,t);          // subtract lsbs, msb: ~borrow-in
	xor_b32(     t,r,a);          // msb: ~borrow-in ^ a
	not_b32(     u,s);            // msb: ~(a^~b)
	and_b32(     s,s,0x80808080); // msb: a ^ ~b
	xor_b32(     r,r,s);          // msb: res = a ^ ~b ^ ~borrow-in
	and_b32(     t,t,u);          // msb: ovfl= (~bw-in ^ a) & ~(a^~b)
	and_b32(     t,t,0x80808080); // ovfl ? 0x80 : 0
	shr_u32(     s,t,7);          // ovfl ? 1 : 0
	not_b32(     u,t);            // ovfl ? 0x7f : 0xff
	and_b32(     r,r,u);          // ovfl ? (a - b) & 0x7f : a - b
	and_b32(     u,a,t);          // ovfl ? a & 0x80 : 0
	sub_u32(     t,t,s);          // ovfl ? 0x7f : 0
	shr_u32(     u,u,7);          // ovfl ? sign(a) : 0
	or_b32(      r,r,t);          // ovfl ? 0x7f : a - b
	add_u32(     r,r,u);          // ovfl ? 0x7f+sign(a) : a - b
	return r;           // byte-wise difference with signed saturation
}

inline unsigned int vsubus4(unsigned int a, unsigned int b)
{
	unsigned int r;
	// This code uses the same basic approach used for the non-saturating 
	// subtraction. The seven low-order bits in each byte are subtracted by 
	// regular subtraction with the inverse of the borrow-out from bit 6 (= 
	// inverse of borrow-in for the msb) being recorded in bit 7, while the 
	// msb is handled separately.
	//
	// Clamping to 0 needs happens when there is a borrow-out from the msb.
	// This is simply accomplished by ANDing the normal addition result with
	// a mask based on the inverted msb borrow-out: ~borrow-out ? 0xff : 0x00.
	// The borrow-out information is generated from the msb. Since we already 
	// have the msb's ~borrow-in and (a^~b) available from the computation of
	// the msb result bit, the most efficient way to compute msb ~borrow-out 
	// is: ((a ^ ~b) & ~borrow-in) | (~b & a). The truth table for the msb is
	//
	// a b bw-in res ~bw-out a^~b (a^~b)&~bw-in (a&~b) ((a^~b)&~bw-in)|(a&~b)
	//                                                        
	// 0 0  0     0     1      1        1          0          1
	// 0 0  1     1     0      1        0          0          0
	// 0 1  0     1     0      0        0          0          0
	// 0 1  1     0     0      0        0          0          0
	// 1 0  0     1     1      0        0          1          1
	// 1 0  1     0     1      0        0          1          1
	// 1 1  0     0     1      1        1          0          1
	// 1 1  1     1     0      1        0          0          0
	//
    
	unsigned int s,t,u;  
	not_b32(  u,b);            // ~b
	xor_b32(  s,u,a);          // a ^ ~b
	and_b32(  u,u,a);          // a & ~b
	or_b32(   r,a,0x80808080); // set msbs
	and_b32(  t,b,0x7f7f7f7f); // clear msbs
	sub_u32(  r,r,t);          // subtract lsbs, msb: ~borrow-in
	and_b32(  t,r,s);          // msb: (a ^ ~b) & ~borrow-in
	and_b32(  s,s,0x80808080); // msb: a ^ ~b
	xor_b32(  r,r,s);          // msb: res = a ^ ~b ^ ~borrow-in
	or_b32(   t,t,u);          // msb: bw-out = ((a^~b)&~bw-in)|(a&~b)
	and_b32(  t,t,0x80808080); // isolate msb: ~borrow-out
	shr_u32(  s,t,7);          // build mask
	sub_u32(  s,t,s);          //  from
	or_b32(   t,t,s);          //   msb
	and_b32(  r,r,t);          // cond. clear result if msb borrow-out
	return r;           // byte-wise difference with unsigned saturation
	}

inline unsigned int vneg4(unsigned int a)
{
	return vsub4 (0, a);// byte-wise negation with wrap-around
}

inline unsigned int vnegss4(unsigned int a)
{
	unsigned int r;
	r = vsub4 (0, a);   //
    
	unsigned int s;      
	and_b32(  a,a,0x80808080); // extract msb
	and_b32(  s,a,r);          // wrap-around if msb set in a and -a
	shr_u32(  s,s,7);          // msb ? 1 : 0
	sub_u32(  r,r,s);          // subtract 1 if result wrapped around
	return r;           // byte-wise negation with signed saturation
}

inline unsigned int vabsdiffs4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	s = vcmpges4 (a, b);// mask = 0xff if a >= b
	r = a ^ b;          //
	s = (r & s) ^ b;    // select a when a >= b, else select b => max(a,b)
	r = s ^ r;          // select a when b >= a, else select b => min(a,b)
	r = vsub4 (s, r);   // |a - b| = max(a,b) - min(a,b);
	return r;           // byte-wise absolute difference of signed integers
}

inline unsigned int vsads4(unsigned int a, unsigned int b)
{
	unsigned int r, s;
	r = vabsdiffs4 (a, b);
	s = r >> 8;
	r = (r & 0x00ff00ff) + (s & 0x00ff00ff);
	r = ((r << 16) + r) >> 16;
	return r;           // byte-wise sum of absolute differences of signed ints
}

#undef not_b32
#undef mov_b32
#undef and_b32
#undef or_b32
#undef xor_b32
#undef shr_u32
#undef shl_u32
#undef sub_u32
#undef add_u32
#undef add_sat_s32
#undef sub_sat_s32
#undef set_ge_s32_s32
#undef set_le_s32_s32
#undef set_gt_s32_s32
#undef set_lt_s32_s32
#undef cvt_s32_s16_s
#undef cvt_s32_s16_u

#endif /* SIMD_FUNCTIONS_H__ */
