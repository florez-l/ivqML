// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <ivqML/Config.h>

// -------------------------------------------------------------------------
#define ivqML_Model_Types                                     \
  using TReal      = typename Superclass::TReal;              \
  using TNatural   = typename Superclass::TNatural;           \
  using TMatrix    = typename Superclass::TMatrix;            \
  using TColumn    = typename Superclass::TColumn;            \
  using TRow       = typename Superclass::TRow;               \
  using TMatrixMap = typename Superclass::TMatrixMap;         \
  using TColumnMap = typename Superclass::TColumnMap;         \
  using TRowMap    = typename Superclass::TRowMap

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _TReal, class _TNatural >
    class Base
    {
    public:
      using Self     = Base;
      using TReal    = _TReal;
      using TNatural = _TNatural;
      using TMatrix  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TColumn  = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
      using TRow     = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

      using TMatrixMap = Eigen::Map< TMatrix >;
      using TColumnMap = Eigen::Map< TColumn >;
      using TRowMap    = Eigen::Map< TRow >;

    public:
      Base( const TNatural& n );
      virtual ~Base( );

      virtual void set_size( const TNatural& n );
      virtual void init( std::function< TReal( ) > g = [](){return( 0 );} );

    protected:
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      TNatural m_S { 0 };
      TReal* m_P   { nullptr };

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Model__Base__h__

// eof - $RCSfile$
