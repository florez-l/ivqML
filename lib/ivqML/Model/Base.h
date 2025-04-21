// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__Base__h__
#define __ivqML__Model__Base__h__

#include <ivqML/Model/Cost.h>

// -------------------------------------------------------------------------
#define ivqML_Model_Types                                     \
  using TReal      = typename Superclass::TReal;              \
  using TNatural   = typename Superclass::TNatural;           \
  using TMatrix    = typename Superclass::TMatrix;            \
  using TColumn    = typename Superclass::TColumn;            \
  using TRow       = typename Superclass::TRow;               \
  using TMatrixMap = typename Superclass::TMatrixMap;         \
  using TColumnMap = typename Superclass::TColumnMap;         \
  using TRowMap    = typename Superclass::TRowMap;            \
  using TCost      = typename Superclass::TCost

// -------------------------------------------------------------------------
namespace ivqML
{
  namespace Model
  {
    /**
     */
    template< class _TReal >
    class Base
    {
    public:
      using Self     = Base;
      using TReal    = _TReal;
      using TNatural = unsigned long long;
      using TMatrix  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TColumn  = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
      using TRow     = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

      using TMatrixMap = Eigen::Map< TMatrix >;
      using TColumnMap = Eigen::Map< TColumn >;
      using TRowMap    = Eigen::Map< TRow >;

      using TCost = ivqML::Model::Cost< TReal >;

    public:
      Base( const TNatural& n );
      virtual ~Base( );

      const TNatural& size( ) const;
      virtual TNatural input_size( ) const = 0;
      virtual TNatural output_size( ) const = 0;
      virtual void set_size( const TNatural& n );
      virtual void init( std::function< TReal( ) > g = [](){return( 0 );} );

      template< class _TG >
      Self& operator+=( const Eigen::EigenBase< _TG >& G );

      template< class _TG >
      Self& operator-=( const Eigen::EigenBase< _TG >& G );

      virtual void allocate_fitting_buffer( const TNatural& M ) const = 0;
      virtual void free_fitting_buffer( ) const = 0;

    protected:
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      TNatural m_S { 0 };
      TReal*   m_P { nullptr };

      TCost m_J;

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#include <ivqML/Model/Base.hxx>

#endif // __ivqML__Model__Base__h__

// eof - $RCSfile$
