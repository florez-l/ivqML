// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __MineSweeperPlayerBase__h__
#define __MineSweeperPlayerBase__h__

/**
 */
class MineSweeperPlayerBase
{
public:
  MineSweeperPlayerBase( );
  virtual ~MineSweeperPlayerBase( ) = default;

  virtual void configure( unsigned int w, unsigned int h, unsigned long n );
  virtual void play( unsigned int& i, unsigned int& j ) = 0;
  virtual void report( const unsigned char& c );

protected:
  unsigned int m_Width;
  unsigned int m_Height;
  unsigned long long m_NumberOfMines;
};

#endif // __MineSweeperPlayerBase__h__

// eof - $RCSfile$
