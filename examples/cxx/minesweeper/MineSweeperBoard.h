// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __MineSweeperBoard__h__
#define __MineSweeperBoard__h__

#include <ostream>
#include <vector>

class MineSweeperPlayerBase;

/**
 */
class MineSweeperBoard
{
public:
  MineSweeperBoard( unsigned int w, unsigned int h, unsigned long n );
  virtual ~MineSweeperBoard( ) = default;

  void set_player( MineSweeperPlayerBase* p );

  const unsigned int& width( ) const;
  const unsigned int& height( ) const;
  const unsigned long long& number_of_mines( ) const;

  bool have_won( ) const;
  bool have_lose( ) const;

  unsigned char click( unsigned int i, unsigned int j );
  void step( );

private:
  void _to_stream( std::ostream& out ) const;
  unsigned long long _idx(
    const unsigned int& i, const unsigned int& j
    ) const;

protected:
  std::vector< bool > m_Patches;
  std::vector< unsigned char > m_Mines;
  unsigned int m_Width { 8 };
  unsigned int m_Height { 8 };
  unsigned long long m_NumberOfMines { 0 };
  bool m_Explosion { false };

  MineSweeperPlayerBase* m_Player;

public:
  friend std::ostream& operator<<( std::ostream& o, const MineSweeperBoard& b )
    {
      b._to_stream( o );
      return( o );
    }
};

#endif // __MineSweeperBoard__h__

// eof - $RCSfile$
